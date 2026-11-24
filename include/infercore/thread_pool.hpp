// InferCore - a small thread pool used by the executor scheduler.
#pragma once

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <functional>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

namespace infercore {

class ThreadPool {
 public:
  explicit ThreadPool(unsigned num_threads);
  ~ThreadPool();

  ThreadPool(const ThreadPool&) = delete;
  ThreadPool& operator=(const ThreadPool&) = delete;

  unsigned size() const { return num_threads_; }

  // Splits [0, n) into chunks of `grain` and runs fn(lo, hi) across the pool.
  // Blocks until all chunks finish.
  void parallel_for(size_t n, size_t grain,
                    const std::function<void(size_t, size_t)>& fn);

  template <typename F>
  void submit(F&& f) {
    {
      std::lock_guard<std::mutex> lk(mutex_);
      tasks_.emplace(std::forward<F>(f));
    }
    cv_.notify_one();
  }

 private:
  void worker_loop();

  unsigned num_threads_;
  std::vector<std::thread> workers_;
  std::queue<std::function<void()>> tasks_;
  std::mutex mutex_;
  std::condition_variable cv_;
  bool stop_ = false;
};

}  // namespace infercore
