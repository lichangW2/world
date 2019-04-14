#ifndef QUEUE_H
#define QUEUE_H

#include <memory>
#include <condition_variable>
#include <mutex>
#include <queue>

/* A simple threadsafe queue using a mutex and a condition variable. */
template <typename T>
class Queue
{
public:
    Queue() = default;

    Queue(Queue&& other)
    {
        std::unique_lock<std::mutex> lock(other.mutex_);
        queue_ = std::move(other.queue_);
    }

    void Push(T value)
    {
        std::unique_lock<std::mutex> lock(mutex_);
        queue_.push(std::move(value));
        lock.unlock();
        cond_.notify_one();
    }

    T Pop()
    {
        std::unique_lock<std::mutex> lock(mutex_);
        cond_.wait(lock, [this]{return !queue_.empty();});
        T value = std::move(queue_.front());
        queue_.pop();
        return value;
    }

    size_t Size()
    {
        std::unique_lock<std::mutex> lock(mutex_);
        return queue_.size();
    }

private:
    mutable std::mutex mutex_;
    std::queue<T> queue_;
    std::condition_variable cond_;
};
#endif // QUEUE_H
