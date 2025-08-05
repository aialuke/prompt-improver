from typing import List, TypeVar, Tuple, Protocol, Any

# Define a protocol for types that support comparison
class SupportsComparison(Protocol):
    def __le__(self, other: Any) -> bool: ...
    def __gt__(self, other: Any) -> bool: ...

# Use the protocol as a type constraint
T = TypeVar('T', bound=SupportsComparison)

def quicksort(arr: List[T], low: int, high: int) -> None:
    """In-place QuickSort with median-of-three pivot selection."""
    def partition(low: int, high: int) -> int:
        # Median-of-three pivot selection
        mid: int = (low + high) // 2
        pivot_candidates: List[Tuple[T, int]] = [
            (arr[low], low),
            (arr[mid], mid),
            (arr[high], high)
        ]
        pivot_value, pivot_idx = sorted(pivot_candidates)[1]
        arr[pivot_idx], arr[high] = arr[high], arr[pivot_idx]
        pivot: T = arr[high]
        i: int = low - 1
        for j in range(low, high):
            if arr[j] <= pivot:
                i += 1
                arr[i], arr[j] = arr[j], arr[i]
        arr[i + 1], arr[high] = arr[high], arr[i + 1]
        return i + 1

    if low < high:
        if high - low <= 10:
            for i in range(low + 1, high + 1):
                key: T = arr[i]
                j: int = i - 1
                while j >= low and arr[j] > key:
                    arr[j + 1] = arr[j]
                    j -= 1
                arr[j + 1] = key
        else:
            pivot_idx: int = partition(low, high)
            quicksort(arr, low, pivot_idx - 1)
            quicksort(arr, pivot_idx + 1, high)

def sort_array(arr: List[T]) -> List[T]:
    """Wrapper for quicksort."""
    quicksort(arr, 0, len(arr) - 1)
    return arr