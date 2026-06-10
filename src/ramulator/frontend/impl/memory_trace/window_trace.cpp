#include <filesystem>
#include <fmt/format.h>
#include <fstream>
#include <iostream>

#include "ramulator/base/param.h"
#include "ramulator/frontend/i_frontend.h"

namespace Ramulator {

namespace fs = std::filesystem;

class WindowTrace : public IFrontEnd, public Implementation {
  RAMULATOR_REGISTER_IMPLEMENTATION(IFrontEnd, WindowTrace, "WindowTrace")

 private:
  struct Trace {
    size_t id;
    Addr_t addr;
    size_t delay;
    bool issue;
  };
  std::vector<std::vector<Trace>> m_trace;

  size_t m_trace_length = 0;
  std::vector<size_t> m_curr_trace_idx;

  size_t m_trace_count = 0;
  std::string m_trace_path;

  size_t m_bank = 0;
  size_t m_queue_len = 0;
  size_t m_rr_offset = 0;   // round-robin 起始 bank，用于公平调度

  std::vector<std::deque<std::pair<bool,size_t>*>> m_read_queue;
  std::vector<size_t> m_stats_wait;
  std::vector<size_t> m_stats_em;
  size_t m_stats_cycle = 0;
  size_t m_issue = 0;

 public:
  void init() override {
    RAMULATOR_PARSE_PARAM(m_clock_ratio, unsigned int, "clock_ratio").required();
    RAMULATOR_PARSE_PARAM(m_trace_path, std::string, "path").required();
    RAMULATOR_PARSE_PARAM(m_bank, size_t, "bank").required();
    RAMULATOR_PARSE_PARAM(m_queue_len, size_t, "queue_len").required();

    for(size_t i = 0; i < m_bank; i++){
      m_read_queue.push_back(std::deque<std::pair<bool, size_t>*>());
      m_curr_trace_idx.push_back(0);
      m_trace.push_back(std::vector<Trace>());
      m_stats_wait.push_back(0);
      m_stats_em.push_back(0);
    }

    m_logger.info(fmt::format("Loading trace file {} ...", m_trace_path));
    init_trace(m_trace_path);
    m_logger.info(fmt::format("Loaded {} banks.", m_trace.size()));
  };

  void tick() override {
    m_stats_cycle++;
    for(size_t i = 0; i < m_bank; i++){
      if(m_read_queue[i].empty()){
        m_stats_em[i]++;
      } else if (m_read_queue[i].front()->second > 1) {
         m_stats_wait[i]++;
      }
      if(!m_read_queue[i].empty() && m_read_queue[i].front()->first){
        if(m_read_queue[i].front()->second <= 1) {
          delete m_read_queue[i].front();
          m_read_queue[i].pop_front();
        } else {
          m_read_queue[i].front()->second--;
        }
      }
    }
    // printf("start issued\n");
    for(size_t k = 0; k < m_bank; k++){
      // 每个周期从 m_rr_offset 开始，环形遍历所有 bank，
      // 避免低编号 bank 永远抢占有限的 send 队列名额。
      size_t i = (m_rr_offset + k) % m_bank;
      if(m_trace[i].empty() || m_curr_trace_idx[i] >= m_trace[i].size()) {
        continue;
      }
      const Trace& t = m_trace[i][m_curr_trace_idx[i]];
      if (m_read_queue[i].size() < m_queue_len) {
        if(t.issue){
          std::pair<bool, size_t>* item = new std::pair<bool, size_t>;
          *item = std::make_pair(false, t.delay);
          Request req(t.addr, Request::Type::Read, -1, [=](Request& req){
            item->first = true;
          });
          req.size_bytes = m_memory_system->get_tx_bytes();
          bool request_sent = m_memory_system->send(req);
          int channel_id = req.addr_vec[0];
          if (request_sent) {
            // printf("issued: %ld, address: 0x%lx, channel_id: %d\n", i, t.addr, channel_id);
            m_read_queue[i].push_back(item);
            m_curr_trace_idx[i] = m_curr_trace_idx[i] + 1;
            m_trace_count++;
            m_issue++;
          } else {
            delete item;
          }
        } else {
          std::pair<bool, size_t>* item = new std::pair<bool, size_t>;
          *item = std::make_pair(true, t.delay);
          m_read_queue[i].push_back(item);
          m_curr_trace_idx[i] = m_curr_trace_idx[i] + 1;
          m_trace_count++;
        }
      }
    }
    // 每个 tick 让起始 bank 向前推进一格，实现公平轮转
    m_rr_offset = (m_rr_offset + 1) % m_bank;
    // printf("end issued\n");
  };

 private:
  // Trace format: one memory access per line, space-separated.
  //   <id> <address> <delay> <issue>
  //
  // - id: bank id
  // - address: memory address (decimal or 0x hex)
  // - delay: delay
  // - issue: issue
  //
  // Example:
  //   0 0x12340 3 0
  //   1 4096 2 1
  //
  // The trace replays cyclically.
  void init_trace(const std::string& file_path_str) {
    fs::path trace_path(file_path_str);
    if (!fs::exists(trace_path)) {
      throw std::runtime_error(fmt::format("Trace {} does not exist!", file_path_str));
    }

    std::ifstream trace_file(trace_path);
    if (!trace_file.is_open()) {
      throw std::runtime_error(fmt::format("Trace {} cannot be opened!", file_path_str));
    }

    std::string line;
    int line_num = 0;
    while (std::getline(trace_file, line)) {
      line_num++;
      std::vector<std::string> tokens;
      tokenize(tokens, line, " ");


      if (tokens.size() != 4) {
        throw std::runtime_error(
            fmt::format("Trace {} line {}: expected 3 tokens, got {}", file_path_str, line_num, tokens.size()));
      }

      
      size_t id = std::stoll(tokens[0]);

      Addr_t addr = -1;
      if (tokens[1].compare(0, 2, "0x") == 0 || tokens[1].compare(0, 2, "0X") == 0) {
        addr = std::stoll(tokens[1].substr(2), nullptr, 16);
      } else {
        addr = std::stoll(tokens[1]);
      }
      
      size_t delay = std::stoll(tokens[2]);
      bool issue = std::stoi(tokens[3]);

      m_trace[id].push_back({id, addr, delay, issue});

      m_trace_length++;
    }

    trace_file.close();
  };

  bool is_finished() override {
    bool finished = m_trace_count >= m_trace_length;
    if(finished) {
      printf("issue: %ld\n", m_issue);
      for (size_t i = 0; i < m_bank;i++){
        float wait = ((float)m_stats_wait[i]) / m_stats_cycle * 100;
        float em = ((float)m_stats_em[i]) / m_stats_cycle * 100;
        printf("Bank: %ld, wait: %f, em: %f, total: %f\n", i, wait, em, wait + em);
      }
    }
    return finished;
  };
};

}  // namespace Ramulator