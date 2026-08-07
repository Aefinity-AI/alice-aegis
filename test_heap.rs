use linked_list_allocator::Heap;
fn test() {
    let mut heap = Heap::empty();
    unsafe { heap.init(0 as *mut u8, 100); }
    unsafe { heap.add_to_heap(1000 as *mut u8, 100); }
}
