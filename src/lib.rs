#![no_std]
/// Custom `Vector` implementation that also supports `#![no_std]`.
use core::alloc::Layout;
use core::fmt::Debug;
use core::ops::{
    Deref, DerefMut, Range, RangeFrom, RangeFull, RangeInclusive, RangeTo, RangeToInclusive,
};
use core::ops::{Index, IndexMut};
use core::ptr::copy_nonoverlapping;
use core::slice::{
    from_raw_parts, from_raw_parts_mut, Chunks, ChunksExact, ChunksExactMut, ChunksMut, Iter,
    IterMut, Windows,
};

#[cfg(feature = "array_windows")]
use core::slice::ArrayWindows;
#[cfg(feature = "array_chunks")]
use core::slice::{ArrayChunks, ArrayChunksMut};
#[cfg(feature = "slice_group_by")]
use core::slice::{GroupBy, GroupByMut};
#[cfg(feature = "slice_concat_trait")]
use alloc::slice::Join;
#[cfg(feature = "slice_concat_trait")]
use alloc::slice::Concat;

extern crate alloc;

/// A custom vector for `#![no_std]` implementation
///
/// This struct will have constant heap allocated value for type [`T`]
/// and would only implement traits for following types:
/// 
/// - [`Index`] (for type [`usize`], [`Range`], [`RangeInclusive`] (and maybe [`isize`]))
/// - [`IndexMut`] (for type [`usize`], [`Range`], [`RangeInclusive`] (and maybe [`isize`]))
/// - [`Iterator`]
/// - [`IntoIterator`]
pub struct Vector<T> {
    /// Pointer to the array
    ptr: *mut T,
    /// Layout defined by the declared type
    /// This contains the total size in bytes with the alignment
    layout: Layout,
    /// Capacity of an array to hold
    capacity: usize,
    /// Actual length of an array.
    len: usize,
}

impl<T> core::fmt::Debug for Vector<T>
where
    T: Debug,
{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        unsafe { core::slice::from_raw_parts(self.ptr, self.len).fmt(f) }
    }
}

impl<T> Vector<T> {
    /// Return total bytes of value, adjusted to multiple of
    /// `core::mem::size_of::<T>()`
    #[inline(always)]
    const fn to_byte_len(val: usize) -> usize {
        val * core::mem::size_of::<T>()
    }

    /// Create a new array, this is a custom made array
    /// with an intention to use `#[no_std]` allocation
    #[inline]
    pub fn new(len: usize) -> Self {
        unsafe {
            let layout = Layout::from_size_align_unchecked(
                Self::to_byte_len(len),
                core::mem::align_of::<T>(),
            );
            Self {
                ptr: alloc::alloc::alloc(layout).cast::<T>(),
                layout,
                len,
                capacity: len,
            }
        }
    }

    /// Create a new array, this is a custom made array
    /// with an intention to use no_std allocation
    #[inline]
    pub fn with_capacity(len: usize) -> Self {
        unsafe {
            let layout = Layout::from_size_align_unchecked(
                Self::to_byte_len(len),
                core::mem::align_of::<T>(),
            );
            Self {
                ptr: alloc::alloc::alloc(layout).cast::<T>(),
                layout,
                len: 0,
                capacity: len,
            }
        }
    }

    /// Creates a new vector from a given slice
    ///
    /// Adds the data one-by-one using push method
    ///
    /// # Example
    ///
    /// ```
    /// use vector::Vector;
    ///
    /// let p: Vector<u32> = Vector::from_slice(&[1, 2, 3, 4, 5, 6]);
    /// let mut it = p.iter();
    ///
    /// assert_eq!(it.next(), Some(&1));
    /// assert_eq!(it.next(), Some(&2));
    /// assert_eq!(it.next(), Some(&3));
    /// assert_eq!(it.next(), Some(&4));
    /// assert_eq!(it.next(), Some(&5));
    /// assert_eq!(it.next(), Some(&6));
    ///
    /// assert_eq!(it.next(), None);
    /// ```
    ///
    #[inline]
    pub fn from_slice(slice: &[T]) -> Self
    where
        T: Copy,
    {
        let mut arr = Self::new(0);
        // Add individually
        slice.iter().for_each(|item| arr.mutate_add(*item));
        arr
    }

    /// Creates a new vector from a given array
    ///
    /// Adds the data individually.
    /// ```
    /// use vector::Vector;
    ///
    /// let mut p: Vector<String> = Vector::from_array([String::from("first"), String::from("second")]);
    /// assert_eq!(p.first(), Some(&String::from("first")));
    /// assert_eq!(p.last(), Some(&String::from("second")));
    /// ```
    #[inline]
    pub fn from_array<const N: usize>(array: [T; N]) -> Self {
        let mut arr = Self::new(0);
        // Add individually
        array.into_iter().for_each(|item| arr.mutate_add(item));
        arr
    }

    /// Mutate and add an element into array.
    ///
    /// Extends an array if the elements exceeds the capacity
    /// of the array.
    ///
    /// ```
    /// use vector::Vector;
    ///
    /// let mut p: Vector<String> = Vector::from_array([String::from("first"), String::from("second")]);
    /// p.push(String::from("third"));
    /// assert_eq!(p.last(), Some(&String::from("third")));
    ///
    /// p.push(String::from("fourth"));
    /// assert_eq!(p.last(), Some(&String::from("fourth")));
    /// ```
    #[inline(always)]
    pub fn push(&mut self, element: T) {
        self.mutate_add(element)
    }

    /// Pops the value from the array.
    ///
    /// Shrinks the capacity if the length is half the capacity
    /// of the array.
    ///
    /// # Example
    ///
    /// ```
    /// use vector::Vector;
    /// let mut vc = Vector::from_array([1, 4, 5, 6, 7, 8, 9, 10]);
    ///
    /// assert_eq!(vc.len(), 8);
    /// assert_eq!(vc.cap(), 8);
    ///
    /// let last_value = vc.pop();
    /// assert_eq!(last_value, Some(10));
    /// assert_eq!(vc.cap(), 8);
    ///
    /// let last_value = vc.pop();
    /// assert_eq!(last_value, Some(9));
    /// assert_eq!(vc.cap(), 8);
    ///
    /// let last_value = vc.pop();
    /// assert_eq!(last_value, Some(8));
    /// assert_eq!(vc.cap(), 8);
    ///
    /// let last_value = vc.pop();
    /// assert_eq!(last_value, Some(7));
    /// assert_eq!(vc.len(), 4);
    /// assert_eq!(vc.cap(), 4);
    ///
    /// let last_value = vc.pop();
    /// assert_eq!(last_value, Some(6));
    /// assert_eq!(vc.len(), 3);
    /// assert_eq!(vc.cap(), 4);
    ///
    /// ```
    #[inline(always)]
    pub fn pop(&mut self) -> Option<T> {
        self.shrink_and_pop()
    }

    fn shrink_and_pop(&mut self) -> Option<T> {
        if self.len == 0 {
            None
        } else {
            let value = unsafe { self.ptr.add(self.len - 1).read() };
            if (self.len - 1) * 2 <= self.capacity {
                let new_layout = unsafe {
                    let new_byte_size = Self::to_byte_len(self.len - 1);
                    Layout::from_size_align_unchecked(new_byte_size, core::mem::align_of::<T>())
                };

                self.ptr = unsafe {
                    // Documentation states that this will be deprecated.
                    let new_ptr = alloc::alloc::alloc_zeroed(new_layout).cast::<T>();
                    copy_nonoverlapping(self.ptr, new_ptr, self.len - 1);
                    alloc::alloc::dealloc(self.ptr.cast::<u8>(), self.layout);
                    new_ptr
                };
                self.layout = new_layout;
                self.capacity = self.len - 1;
            }

            self.len -= 1;
            Some(value)
        }
    }

    /// Mutate and add an element into array to fit the size internally.
    fn mutate_add(&mut self, element: T) {
        if self.len == 0 {
            *self = Self::zeroed(1);
            unsafe { self.ptr.write(element) };
        } else {
            if self.len == self.capacity {
                // Allocate 2 times the current size of memory.
                // A new layout is created here
                let new_layout = unsafe {
                    let new_byte_size = Self::to_byte_len(self.len * 2);
                    Layout::from_size_align_unchecked(new_byte_size, core::mem::align_of::<T>())
                };

                self.ptr = unsafe {
                    // Documentation states that this will be deprecated.
                    let new_ptr = alloc::alloc::alloc_zeroed(new_layout).cast::<T>();
                    copy_nonoverlapping(self.ptr, new_ptr, self.len);
                    alloc::alloc::dealloc(self.ptr.cast::<u8>(), self.layout);
                    new_ptr
                };
                self.layout = new_layout;
                self.capacity = self.len * 2;
            }
            // Finally write to the pointer
            unsafe { self.ptr.add(self.len).write(element) };
            self.len += 1;
        }
    }

    /// Mutates the capacity by extra capacity. Also considers the excess
    /// the [`self`] has and increases appropriately
    #[inline(always)]
    pub fn mutate_capacity_by(&mut self, extra_capacity: usize) {
        // If the extra capacity requested is greater then no need
        // of extending the array.
        if self.capacity < self.len + extra_capacity {
            // Remove the excess capacity
            self.mutate_capacity_by_ignore_current(extra_capacity - (self.capacity - self.len));
        }
    }

    /// Mutates the capacity of array
    ///
    /// ## Note
    /// This does not take into consideration the current
    /// capacity and directly increases by [`extra_capacity`].
    /// To extend capacity appropriately, refer function [`mutate_capacity_by`]
    ///
    /// ## Safety
    /// Reallocates the pointer with new size and copies all the content from
    /// old location
    pub fn mutate_capacity_by_ignore_current(&mut self, extra_capacity: usize) {
        if self.len == 0 {
            *self = Self::zeroed(extra_capacity);
        } else {
            // Allocate precise extra capacity, with extra space for alignment
            let new_layout = unsafe {
                let new_byte_size = Self::to_byte_len(self.capacity + extra_capacity);
                Layout::from_size_align_unchecked(new_byte_size, 0x08)
            };

            self.ptr = unsafe {
                // Documentation states that this will be deprecated.
                let new_ptr = alloc::alloc::alloc_zeroed(new_layout).cast::<T>();
                copy_nonoverlapping(self.ptr, new_ptr, self.len);
                alloc::alloc::dealloc(self.ptr.cast::<u8>(), self.layout);
                new_ptr
            };
            self.layout = new_layout;
            self.capacity += extra_capacity;
        }
    }

    /// Extend from an slice with size hint.
    ///
    /// ## Performance
    /// If multiple block of elements (or small array) are being inserted
    /// using iterators, then use method [`push`] instead
    #[inline]
    pub fn extend_from_slice(&mut self, slice: &[T])
    where
        T: Copy,
    {
        let lower_bound = slice.len();
        // Mutate capacity by certain at least `lower_bound`
        self.mutate_capacity_by(lower_bound);
        // Reallocation can be done if extra iter needed
        slice.iter().for_each(|f| self.mutate_add(*f));
    }

    /// Extend from an iterator with size hint.
    ///
    /// ## Performance
    /// If multiple block of elements (or small array) are being inserted
    /// using iterators, then use method [`push`] instead
    #[inline]
    pub fn extend_from_iter(&mut self, other_iter: &mut Iter<'_, T>)
    where
        T: Copy,
    {
        let (lower_bound, _) = other_iter.size_hint();
        // Mutate capacity by certain at least `lower_bound`
        self.mutate_capacity_by(lower_bound);
        // Reallocation can be done if extra iter needed
        other_iter.for_each(|f| self.mutate_add(*f));
    }

    /// Length of the array
    #[inline(always)]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Length of the array
    #[inline(always)]
    pub const fn len(&self) -> usize {
        self.len
    }

    /// Maximum element that the array can hold; after which adding
    /// more elements will mutate the capacity two-folds
    #[inline(always)]
    pub const fn cap(&self) -> usize {
        self.capacity
    }

    /// Returns the borrowed values from iterator
    #[inline(always)]
    pub fn iter(&self) -> Iter<'_, T> {
        unsafe { from_raw_parts(self.ptr, self.len).iter() }
    }

    /// Returns the mutable borrowed iterator
    #[inline(always)]
    pub fn iter_mut(&mut self) -> IterMut<'_, T> {
        unsafe { from_raw_parts_mut(self.ptr, self.len).iter_mut() }
    }

    /// Returns the chunks of `n` size
    ///
    /// Remainder will yield array of `len < chunk_size`
    #[inline(always)]
    pub fn chunks(&self, chunk_size: usize) -> Chunks<'_, T> {
        unsafe { from_raw_parts(self.ptr, self.len).chunks(chunk_size) }
    }

    /// Returns the chunks of exactly `n` size
    ///
    /// Remainder can be accessed from `remainder` method from
    /// iterator [`ChunkExact`]
    #[inline(always)]
    pub fn chunks_exact(&self, chunk_size: usize) -> ChunksExact<'_, T> {
        unsafe { from_raw_parts(self.ptr, self.len).chunks_exact(chunk_size) }
    }

    /// Returns the mutable slice of `n` size
    ///
    /// Remainder will yield array of `len < chunk_size`
    #[inline(always)]
    pub fn chunks_mut(&self, chunk_size: usize) -> ChunksMut<'_, T> {
        unsafe { from_raw_parts_mut(self.ptr, self.len).chunks_mut(chunk_size) }
    }

    /// Returns the mutable chunks of exactly `n` size
    ///
    /// Remainder can be accessed from `remainder` from
    /// iterator [`ChunkExactMut`]
    #[inline(always)]
    pub fn chunks_exact_mut(&self, chunk_size: usize) -> ChunksExactMut<'_, T> {
        unsafe { from_raw_parts_mut(self.ptr, self.len).chunks_exact_mut(chunk_size) }
    }

    /// Returns the window iterator of exactly `n` size, iterating through
    /// array.
    ///
    /// If window size is larger than the slice, then it won't yield a slice.
    #[inline(always)]
    pub fn windows(&self, window_size: usize) -> Windows<'_, T> {
        unsafe { from_raw_parts_mut(self.ptr, self.len).windows(window_size) }
    }

    /// Returns the window iterator of exactly `n` size, iterating through
    /// array with single step
    ///
    /// If window size is larger, then it won't yield a slice.
    #[inline(always)]
    #[cfg(feature = "array_chunks")]
    pub fn array_chunks<const SIZE: usize>(&self) -> VectorChunks<'_, T, SIZE> {
        unsafe { from_raw_parts_mut(self.ptr, self.len).array_chunks::<SIZE>() }
    }

    /// Returns the window iterator of exactly `n` size, iterating through
    /// array with single step
    ///
    /// If window size is larger, then it won't yield a slice.
    #[inline(always)]
    #[cfg(feature = "array_chunks")]
    pub fn array_chunks_mut<const SIZE: usize>(&self) -> VectorChunksMut<'_, T, SIZE> {
        unsafe { from_raw_parts_mut(self.ptr, self.len).array_chunks_mut::<SIZE>() }
    }

    /// Returns the static array window of exactly size `SIZE`
    #[inline(always)]
    #[cfg(feature = "array_windows")]
    pub fn array_windows<const SIZE: usize>(&self) -> VectorWindows<'_, T, SIZE> {
        unsafe { from_raw_parts_mut(self.ptr, self.len).array_windows::<SIZE>() }
    }

    /// Returns the slice group that separates the values based on condition
    /// defined by [`f`]
    #[inline(always)]
    #[cfg(feature = "slice_group_by")]
    pub fn group_by<'a, F>(&self, f: F) -> GroupBy<'a, T, F>
    where
        F: FnMut(&T, &T) -> bool,
    {
        unsafe { from_raw_parts_mut(self.ptr, self.len).group_by::<F>(f) }
    }

    /// Returns the mutable slice group that separates the values based on condition
    #[inline(always)]
    #[cfg(feature = "slice_group_by")]
    pub fn group_by_mut<'a, F>(&self, f: F) -> GroupByMut<'a, T, F>
    where
        F: FnMut(&T, &T) -> bool,
    {
        unsafe { from_raw_parts_mut(self.ptr, self.len).group_by_mut::<F>(f) }
    }

    /// Create a new array filled with zero
    ///
    /// # Example
    ///
    /// ```
    /// use vector::Vector;
    ///
    /// let vector: Vector<u16> = Vector::zeroed(12);
    /// assert!(vector.iter().all(|item| *item == 0));
    /// ```
    #[inline]
    pub fn zeroed(len: usize) -> Self {
        unsafe {
            let layout = Layout::from_size_align_unchecked(
                Self::to_byte_len(len),
                core::mem::align_of::<T>(),
            );
            Self {
                ptr: alloc::alloc::alloc_zeroed(layout).cast::<T>(),
                layout,
                len,
                capacity: len,
            }
        }
    }
}

impl<T> PartialEq for Vector<T>
where
    T: PartialEq,
{
    #[inline(always)]
    fn eq(&self, other: &Self) -> bool {
        self.len == other.len && self.iter().zip(other).all(|(a, b)| a == b)
    }
}

impl<T> PartialEq<[T]> for Vector<T>
where
    T: PartialEq,
{
    #[inline(always)]
    fn eq(&self, other: &[T]) -> bool {
        self.len == other.len() && self.iter().zip(other).all(|(a, b)| a == b)
    }
}

impl<T> PartialEq<&[T]> for Vector<T>
where
    T: PartialEq,
{
    #[inline(always)]
    fn eq(&self, other: &&[T]) -> bool {
        self.len == other.len() && self.iter().zip(*other).all(|(a, b)| a == b)
    }
}

impl<T> FromIterator<T> for Vector<T> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        // Newly created array, with size zero
        let mut new_array: Vector<T> = Vector::zeroed(0);

        // Unfortunately this is extended similar to [`std::vec::Vec`]
        // and returned to the user, as there is no hint in advance the
        // total elements returned by the iterator
        for item in iter {
            new_array.mutate_add(item);
        }
        // A space optimization could be to
        // apply shrink-to-fit, but this is not the focus; currently.
        new_array
    }
}

impl<'a, T> IntoIterator for &'a Vector<T> {
    // Into Iterator for &'a Vector<T>
    type Item = &'a T;
    type IntoIter = Iter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<T> Index<usize> for Vector<T> {
    type Output = T;

    #[inline]
    fn index(&self, index: usize) -> &Self::Output {
        debug_assert!(index < self.len);
        unsafe { &*self.ptr.add(index) }
    }
}

impl<T> Deref for Vector<T> {
    type Target = [T];
    /// Dereference for returning the whole slice
    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        &self[..]
    }
}

impl<T> DerefMut for Vector<T> {
    /// Mutable dereference for returning the whole slice
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self[..]
    }
}

impl<T> Index<Range<usize>> for Vector<T> {
    type Output = [T];

    #[inline]
    fn index(&self, range: Range<usize>) -> &Self::Output {
        debug_assert!(range.start < self.len && range.end <= self.len);
        unsafe { from_raw_parts(self.ptr.add(range.start), range.end - range.start) }
    }
}

impl<T> Index<RangeFrom<usize>> for Vector<T> {
    type Output = [T];

    /// Returns slice starting from certain starting point till the end
    ///
    /// # Panics
    /// Panics if starting point is greater than the length.
    #[inline]
    fn index(&self, range: RangeFrom<usize>) -> &Self::Output {
        // UNSAFE: Only way for this is to expose from raw parts
        debug_assert!(range.start < self.len);
        unsafe { from_raw_parts(self.ptr.add(range.start), self.len - range.start) }
    }
}

impl<T> Index<RangeInclusive<usize>> for Vector<T> {
    type Output = [T];

    #[inline]
    fn index(&self, range: RangeInclusive<usize>) -> &Self::Output {
        debug_assert!(*range.start() < self.len && *range.end() < self.len);
        unsafe { from_raw_parts(self.ptr.add(*range.start()), *range.end() - *range.start()) }
    }
}

impl<T> Index<RangeTo<usize>> for Vector<T> {
    type Output = [T];
    #[inline]
    fn index(&self, range: RangeTo<usize>) -> &Self::Output {
        // UNSAFE: Only way for this is to expose from raw parts
        debug_assert!(range.end < self.len);
        unsafe { from_raw_parts(self.ptr, range.end) }
    }
}

impl<T> Index<RangeToInclusive<usize>> for Vector<T> {
    type Output = [T];

    #[inline]
    fn index(&self, range: RangeToInclusive<usize>) -> &Self::Output {
        // UNSAFE: Only way for this is to expose from raw parts
        debug_assert!(range.end < self.len);
        unsafe { from_raw_parts(self.ptr, range.end) }
    }
}

impl<T> Index<RangeFull> for Vector<T> {
    type Output = [T];

    #[inline(always)]
    fn index(&self, _: RangeFull) -> &Self::Output {
        // UNSAFE: Only way for this is to expose from raw parts
        unsafe { from_raw_parts(self.ptr, self.len) }
    }
}

impl<T> IndexMut<Range<usize>> for Vector<T> {
    #[inline]
    fn index_mut(&mut self, range: Range<usize>) -> &mut [T] {
        // UNSAFE: Only way for this is to expose from raw parts
        debug_assert!(range.start < self.len && range.end < self.len);
        unsafe { &mut *from_raw_parts_mut(self.ptr.add(range.start), range.end - range.start) }
    }
}

impl<T> IndexMut<RangeFrom<usize>> for Vector<T> {
    #[inline]
    fn index_mut(&mut self, range: RangeFrom<usize>) -> &mut [T] {
        // UNSAFE: Only way for this is to expose from raw parts
        debug_assert!(range.start < self.len);
        unsafe { &mut *from_raw_parts_mut(self.ptr.add(range.start), self.len - range.start) }
    }
}

impl<T> IndexMut<RangeInclusive<usize>> for Vector<T> {
    #[inline]
    fn index_mut(&mut self, range: RangeInclusive<usize>) -> &mut [T] {
        // UNSAFE: Only way for this is to expose from raw parts
        debug_assert!(*range.start() < self.len && *range.end() < self.len);
        unsafe {
            &mut *from_raw_parts_mut(self.ptr.add(*range.start()), *range.end() - *range.start())
        }
    }
}

impl<T> IndexMut<RangeTo<usize>> for Vector<T> {
    #[inline]
    fn index_mut(&mut self, range: RangeTo<usize>) -> &mut [T] {
        // UNSAFE: Only way for this is to expose from raw parts
        debug_assert!(range.end < self.len);
        unsafe { &mut *from_raw_parts_mut(self.ptr, range.end) }
    }
}

impl<T> IndexMut<RangeToInclusive<usize>> for Vector<T> {
    #[inline]
    fn index_mut(&mut self, range: RangeToInclusive<usize>) -> &mut [T] {
        // UNSAFE: Only way for this is to expose from raw parts
        debug_assert!(range.end < self.len);
        unsafe { &mut *from_raw_parts_mut(self.ptr, range.end) }
    }
}

impl<T> IndexMut<RangeFull> for Vector<T> {
    /// Returns entire mutable slice of the array
    #[inline(always)]
    fn index_mut(&mut self, _: RangeFull) -> &mut [T] {
        // UNSAFE: Only way for this is to expose from raw parts
        unsafe { &mut *from_raw_parts_mut(self.ptr, self.len) }
    }
}

// impl core::

impl<T> IndexMut<usize> for Vector<T> {
    /// Returns mutable index to the user
    #[inline]
    fn index_mut(&mut self, index: usize) -> &mut T {
        debug_assert!(index < self.len);
        unsafe { &mut *self.ptr.add(index) }
    }
}

impl<T> Drop for Vector<T> {
    /// A simple deallocation for pointer.
    #[inline]
    fn drop(&mut self) {
        // UNSAFE: Deallocate the pointer, no other handling to perform
        unsafe { alloc::alloc::dealloc(self.ptr.cast::<u8>(), self.layout) }
    }
}

#[cfg(feature = "slice_concat_trait")]
impl<T: Copy> Join<T> for Vector<Vector<T>> {
    type Output = Vector<T>;

    fn join(slice: &Self, sep: T) -> Self::Output {
        let first = match slice.first() {
            Some(vec) => vec,
            None => return Vector::new(0)
        };
        let size = slice.iter().skip(1).fold(first.len(), |prev, curr| prev + 1 + curr.len());

        let mut result = Vector::with_capacity(size);
        result.extend_from_slice(first);

        slice.iter().skip(1).for_each(|sl| {
            result.push(sep);
            result.extend_from_slice(sl);
        });

        result
    }
}

#[cfg(feature = "slice_concat_trait")]
impl<T: Copy> Join<T> for Vector<&[T]> {
    type Output = Vector<T>;

    fn join(slice: &Self, sep: T) -> Self::Output {
        let first = match slice.first() {
            Some(vec) => vec,
            None => return Vector::new(0)
        };
        let size = slice.iter().skip(1).fold(first.len(), |prev, curr| prev + 1 + curr.len());

        let mut result = Vector::with_capacity(size);
        result.extend_from_slice(first);

        slice.iter().skip(1).for_each(|sl| {
            result.push(sep);
            result.extend_from_slice(sl);
        });

        result
    }
}

#[cfg(feature = "slice_concat_trait")]
impl<T> Join<[T]> for Vector<Vector<T>> where T: Copy, [T]: Sized {
    type Output = Vector<T>;

    fn join(slice: &Self, sep: [T]) -> Self::Output {
        let first = match slice.first() {
            Some(vec) => vec,
            None => return Vector::new(0)
        };
        let size = slice.iter().skip(1).fold(first.len(), |prev, curr| prev + 1 + curr.len());

        let mut result = Vector::with_capacity(size);
        result.extend_from_slice(first);

        slice.iter().skip(1).for_each(|sl| {
            result.extend_from_slice(&sep);
            result.extend_from_slice(sl);
        });

        result
    }
}

#[cfg(feature = "slice_concat_trait")]
impl<T: Copy> Concat<T> for Vector<Vector<T>> {
    type Output = Vector<T>;

    fn concat(slice: &Self) -> Self::Output {
        let first = match slice.first() {
            Some(vec) => vec,
            None => return Vector::new(0)
        };
        let size = slice.iter().skip(1).fold(first.len(), |prev, curr| prev + curr.len());

        let mut result = Vector::with_capacity(size);
        result.extend_from_slice(first);

        slice.iter().skip(1).for_each(|sl| {
            result.extend_from_slice(sl);
        });

        result
    }
}

#[cfg(feature = "slice_concat_trait")]
impl<T: Copy> Concat<T> for Vector<&[T]> {
    type Output = Vector<T>;

    fn concat(slice: &Self) -> Self::Output {
        let first = match slice.first() {
            Some(vec) => vec,
            None => return Vector::new(0)
        };
        let size = slice.iter().skip(1).fold(first.len(), |prev, curr| prev + curr.len());

        let mut result = Vector::with_capacity(size);
        result.extend_from_slice(first);

        slice.iter().skip(1).for_each(|sl| {
            result.extend_from_slice(sl);
        });

        result
    }
}

#[cfg(feature = "slice_concat_trait")]
impl<T> Concat<[T]> for Vector<[T]> where T: Copy, [T]: Sized {
    type Output = Vector<T>;

    fn concat(slice: &Self) -> Self::Output {
        let first = match slice.first() {
            Some(vec) => vec,
            None => return Vector::new(0)
        };
        let size = slice.iter().skip(1).fold(first.len(), |prev, curr| prev + curr.len());

        let mut result = Vector::with_capacity(size);
        result.extend_from_slice(first);

        slice.iter().skip(1).for_each(|sl| {
            result.extend_from_slice(sl);
        });

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    pub fn test_alloc() {
        let mut p: Vector<i32> = Vector::new(5);
        p[0] = 1;
        p[1] = 3;
        p[2] = 4;
        p[3] = 6;
        p[4] = 1 << 30;

        assert_eq!(p[0], 1);
        assert_eq!(p[1], 3);
        assert_eq!(p[2], 4);
        assert_eq!(p[3], 6);
        assert_eq!(p[4], 1 << 30);
    }

    #[test]
    pub fn test_iter() {
        let en = 65536;
        let p: Vector<usize> = (0..en).collect();

        assert!(p.iter().zip(0..65536).all(|(a, b)| *a == b));
    }

    #[test]
    pub fn filter_test() {
        let mut new_arr: Vector<u32> = (0..100).collect();
        let vec: Vector<u32> = (0..65536).collect();
        new_arr.extend_from_iter(&mut vec.iter());

        assert_eq!(new_arr.len(), 100 + 65536);
        // According to array alignment constraints, nearest big
        // value divisble by 32 is the below number.
        assert_eq!(new_arr.cap(), 65536 + 100);

        let another_arr: Vector<u32> = (0..1928).collect();
        new_arr.extend_from_iter(&mut another_arr.iter());
        assert_eq!(new_arr.len(), 100 + 1928 + 65536);
        // According to array alignment constraints, nearest big
        // value divisble by 32 is the below number.
        assert_eq!(new_arr.cap(), 100 + 1928 + 65536);

        assert!(new_arr
            .iter()
            .zip((0..100).chain(0..65536).chain(0..1928))
            .all(|(a, b)| *a == b));
    }

    #[test]
    pub fn test_mutate_add_small_arr() {
        // Collect uses mutate_add: therefore, increases capacity 2 times
        // if exceeds, but base minimum capacity is `32`.
        let new_arr: Vector<u32> = (0..8).collect();
        assert_eq!(new_arr.len(), 8);
        assert_eq!(new_arr.cap(), 8);
    }

    #[test]
    pub fn test_mutate_add() {
        // Collect uses mutate_add: therefore, increases capacity 2 times
        // if exceeds, but base size is 32.
        let mut new_arr: Vector<u32> = (0..100).collect();
        assert_eq!(new_arr.len(), 100);
        assert_eq!(new_arr.cap(), 128);

        // This loop forces the addition to extend capacity to
        // Twice the original one.
        for x in 100..200 {
            new_arr.push(x);
        }

        assert!(new_arr.iter().zip(0..200).all(|(a, b)| *a == b));
        assert_eq!(new_arr.len(), 200);
        assert_eq!(new_arr.cap(), 256);
    }

    #[test]
    fn test_for_loop() {
        let p: Vector<usize> = (0..100).step_by(2).collect();
        let mut start_value = 0;
        for x in &p {
            assert_eq!(start_value, *x);
            start_value += 2;
        }
    }
}
