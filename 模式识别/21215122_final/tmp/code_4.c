static void pool_release_buffer(void *opaque, uint8_t *data)

{

    BufferPoolEntry *buf = opaque;

    AVBufferPool *pool = buf->pool;



    if(CONFIG_MEMORY_POISONING)

        memset(buf->data, 0x2a, pool->size);



    add_to_pool(buf);

    if (!avpriv_atomic_int_add_and_fetch(&pool->refcount, -1))

        buffer_pool_free(pool);

}
