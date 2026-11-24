#include "infercore/thread_pool.hpp"

#include <algorithm>

namespace infercore {

ThreadPool::ThreadPool(unsigned num_threads) : num_threads_(num_threads) {
  if (num_threads_ == 0) num_threads_ = 1;
  workers_.reserve(num_threads_);
  for (unsigned i = 0; i < num_threads_; ++i)
    workers_.emplace_back([this] { worker_loop(); });
}

ThreadPool::~ThreadPool() {
  {
    std::lock_guard<std::mutex> lk(mutex_);
    stop_ = true;
  }
  cv_.notify_all();
  for (auto& w : workers_) w.join();
}

void ThreadPool::worker_loop() {
  for (;;) {
    std::function<void()> task;
    {
      std::unique_lock<std::mutex> lk(mutex_);
      cv_.wait(lk, [&] { return stop_ || !tasks_.empty(); });
      if (stop_ && tasks_.empty()) return;
      task = std::move(tasks_.front());
      tasks_.pop();
    }
    task();
  }
}

void ThreadPool::parallel_for(size_t n, size_t grain,
                              const std::function<void(size_t, size_t)>& fn) {
  if (n == 0) return;
  if (grain == 0) grain = 1;
  const size_t nchunks = (n + grain - 1) / grain;
  if (num_threads_ <= 1 || nchunks <= 1) {
    fn(0, n);
    return;
  }
  std::atomic<size_t> remaining{nchunks};
  std::mutex done_m;
  std::condition_variable done_cv;
  for (size_t c = 0; c < nchunks; ++c) {
    const size_t lo = c * grain;
    const size_t hi = std::min(n, lo + grain);
    submit([&, lo, hi] {
      fn(lo, hi);
      if (remaining.fetch_sub(1, std::memory_order_acq_rel) == 1) {
        std::lock_guard<std::mutex> g(done_m);
        done_cv.notify_one();
      }
    });
  }
  std::unique_lock<std::mutex> lk(done_m);
  done_cv.wait(lk, [&] { return remaining.load(std::memory_order_acquire) == 0; });
}

}  // namespace infercore
