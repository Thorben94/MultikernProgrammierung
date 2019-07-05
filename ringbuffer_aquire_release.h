#include <atomic>
#include <array>

template<class T> struct bufferedElem
{
  T* elem;
  size_t padding[32];

};

/* Ringbuffer holds S pointers to type T. */
template<class T, size_t S> class SCSPRingBuffer
{
	private:
	/* Proposed data layout. */
	std::array<bufferedElem<T>, S> buffer{};
	std::atomic_size_t head = 0;
    size_t padding[64];
    std::atomic_size_t tail = 0;
	
	public:

	/* Add pointer to the end of the buffer and return true.
	Return false if the buffer is full. */
	bool tryPush(T* pointer)
	{
      size_t myhead = head.load(std::memory_order_relaxed);
		if ((myhead + 1) % S == tail.load(std::memory_order_acquire))
		{
			return false;
		}
		else	// my rationale is that this else ensures sequential ordering, but i'm not certain
		{
			buffer[myhead].elem = pointer;
            head.store((1 + myhead) % S, std::memory_order_release);
            return true;
		}
	}

	/* Return and remove first pointer in the buffer.
	Return nullptr if the buffer is empty. */
	T* tryPop()
	{
      size_t mytail = tail.load(std::memory_order_relaxed);
		if (mytail == head.load(std::memory_order_acquire)) 	
		{
			return nullptr;
		}
		else
		{
			T* myt = buffer[mytail].elem;
            tail.store((1 + mytail) % S, std::memory_order_release);
            return myt;
		}
	}
};