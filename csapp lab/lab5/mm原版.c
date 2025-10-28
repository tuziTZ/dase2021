/*
 * mm-naive.c - The fastest, least memory-efficient malloc package.
 * 
 * In this naive approach, a block is allocated by simply incrementing
 * the brk pointer.  A block is pure payload. There are no headers or
 * footers.  Blocks are never coalesced or reused. Realloc is
 * implemented directly using mm_malloc and mm_free.
 *
 * NOTE TO STUDENTS: Replace this header comment with your own header
 * comment that gives a high level description of your solution.
 */
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <unistd.h>
#include <string.h>

#include "mm.h"
#include "memlib.h"

/*********************************************************
 * NOTE TO STUDENTS: Before you do anything else, please
 * provide your team information in the following struct.
 ********************************************************/
team_t team = {
    /* Team name */
    "10215501412",
    /* First member's full name */
    "Harry Bovik",
    /* First member's email address */
    "bovik@cs.cmu.edu",
    /* Second member's full name (leave blank if none) */
    "",
    /* Second member's email address (leave blank if none) */
    ""
};


#define ALIGNMENT 8
#define ALIGN(size) (((size) + (ALIGNMENT-1)) & ~0x7)
#define SIZE_T_SIZE (ALIGN(sizeof(size_t)))

#define WSIZE 4 
#define DSIZE 8 
#define CHUNKSIZE (1<<12) //初始空闲块的大小和扩展堆时的默认大小

#define MAX(x,y) ((x) > (y)? (x) : (y))

/*将块的大小和已分配位|起来，存放在头部或脚部*/
#define PACK(size,alloc) ((size)|(alloc))

/*从地址p读或者写一个四字节的数*/
#define GET(p) (*(unsigned int *) (p))
#define PUT(p,val) (*(unsigned int *) (p)=(val))

/*当p指向一个头部或者脚部时，获取该部分所标记的块大小或已分配位*/
#define GET_SIZE(p) (GET(p) & ~0x7)
#define GET_ALLOC(p) (GET(p) & 0x1)

/*当bp指向一个块时，获得头部地址或者脚部地址*/
#define HDRP(bp) ((char *)(bp) - WSIZE)//有效载荷块向前4位就是头部
#define FTRP(bp) ((char *)(bp) + GET_SIZE(HDRP(bp)) - DSIZE)//DSIZE的大小是头部和脚部相加的大小，所以要减去                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                

/*当bp指向一个块时，获得后一个块地址或者前一个块地址*/
#define NEXT_BLKP(bp) ((char *)(bp) + GET_SIZE(((char *)(bp) - WSIZE)))//GET_SIZE取到了bp指向块的大小
#define PREV_BLKP(bp) ((char *)(bp) - GET_SIZE(((char *)(bp) - DSIZE)))//GET_SIZE取到了bp前一个块的大小

static char *heap_listp=0;//堆的起始位置NULL

/* 合并前后的空闲块 */
static void *coalesce(void *bp)
{
	/* 获取前后两个块的空闲情况 */
	size_t prev_alloc = GET_ALLOC(FTRP(PREV_BLKP(bp)));
	size_t next_alloc = GET_ALLOC(HDRP(NEXT_BLKP(bp)));
	size_t size = GET_SIZE(HDRP(bp));//获取当前块的大小
	
	if (prev_alloc && next_alloc) {				/* 前后都已分配 */
		return bp;
	} else if (prev_alloc && !next_alloc) {		/* 后一个块空闲 */
		size += GET_SIZE(HDRP(NEXT_BLKP(bp)));
		PUT(HDRP(bp), PACK(size, 0));
		PUT(FTRP(bp), PACK(size, 0));
	} else if (!prev_alloc && next_alloc) {		/* 前一个块空闲 */
		size += GET_SIZE(FTRP(PREV_BLKP(bp)));
		PUT(FTRP(bp), PACK(size, 0));
		PUT(HDRP(PREV_BLKP(bp)), PACK(size, 0));
		bp = PREV_BLKP(bp);
	} else if (!prev_alloc && !next_alloc) {	/* 前后都空闲 */
		size += GET_SIZE(HDRP(PREV_BLKP(bp))) 
			+ GET_SIZE(FTRP(NEXT_BLKP(bp)));
		PUT(HDRP(PREV_BLKP(bp)), PACK(size, 0));
		PUT(FTRP(NEXT_BLKP(bp)), PACK(size, 0));
		bp = PREV_BLKP(bp);
	}
	
	return bp;
}

/* 用一个新的空闲块扩展堆 */
static void *extend_heap(size_t words) 
{
	char *bp;
	size_t size;
	
	/* 分配偶数个字，保证八字节对齐 */
	size=(words%2) ? (words+1)*WSIZE : words*WSIZE;
	if((long) (bp=mem_sbrk(size))==-1)//申请的区域超出最大堆的大小
		return NULL;
	/* 添加空闲块的头部和脚部 */
	PUT(HDRP(bp),PACK(size,0));
	PUT(FTRP(bp),PACK(size,0));
	PUT(HDRP(NEXT_BLKP(bp)),PACK(0,1));
	
	/*采用立即边界合并方式：在每次一个块被释放时就合并所有的相邻块*/
	return coalesce(bp);
}


/* 从头到尾遍历隐式链表，找到可以分配的空闲块 */
static void *find_fit(size_t asize)
{
	void* p;

	for (p = heap_listp; GET_SIZE(HDRP(p)) > 0; p = NEXT_BLKP(p)) {
		if (!GET_ALLOC(HDRP(p)) && (asize <= GET_SIZE(HDRP(p))))
			return p;
	}
	
	return NULL;
}

/* 在空闲块头部分配一个块，使满足八字节对齐，如果分配块小于最小块的大小，就不进行分割 */
static void place(void *bp, size_t asize)
{
	size_t size = GET_SIZE(HDRP(bp));
	if (size - asize >= 2*WSIZE) {
		PUT(HDRP(bp), PACK(asize, 1));
		PUT(FTRP(bp), PACK(asize, 1));
		bp = NEXT_BLKP(bp);
		PUT(HDRP(bp), PACK(size-asize, 0));
		PUT(FTRP(bp), PACK(size-asize, 0));
	} else {
		PUT(HDRP(bp), PACK(size, 1));
		PUT(FTRP(bp), PACK(size, 1));
	}
}




/* 
 * mm_init - initialize the malloc package.
 */
int mm_init(void)
{
    /* 创建一个最小块，包含序言，头部，脚部，结尾块 */
    if ((heap_listp = mem_sbrk(4*WSIZE)) == (void *)-1)
    	return -1;
    
    PUT(heap_listp, 0);								
    PUT(heap_listp + (1*WSIZE), PACK(DSIZE, 1));	
    PUT(heap_listp + (2*WSIZE), PACK(DSIZE, 1));	
    PUT(heap_listp + (3*WSIZE), PACK(0, 1));		
    heap_listp += (2*WSIZE);
    
    if (extend_heap(CHUNKSIZE/WSIZE) == NULL) {
    	return -1;
   	}

    return 0;
}





/* 
 * mm_malloc - Allocate a block by incrementing the brk pointer.
 *     Always allocate a block whose size is a multiple of the alignment.
 */
void *mm_malloc(size_t size)
{
    size_t asize;		
    size_t extendsize;	/* 如果要扩展堆，所扩展的字节数 */
   	char *bp;
   	
   	/* size=0时不分配 */
   	if (size == 0)
   		return NULL;
   	
   	/* 使asize满足8字节对齐 */
   	if (size <= DSIZE)
   		asize = 2*DSIZE;
   	else 
   		asize = DSIZE * ((size + (DSIZE) + (DSIZE-1)) / DSIZE);
   	
   	/* 顺序查找链表 */
   	if ((bp = find_fit(asize)) != NULL) {
   		place(bp, asize);
   		return bp;
   	}
   	
   	/* 扩展堆 */
   	extendsize = MAX(asize, CHUNKSIZE);
   	if ((bp = extend_heap(extendsize/WSIZE)) == NULL)
   		return NULL;
   	place(bp, asize);

   	return bp;
}



/*
 * mm_free - Freeing a block does nothing.
 */
void mm_free(void *ptr)
{
	size_t size = GET_SIZE(HDRP(ptr));
	
	PUT(HDRP(ptr), PACK(size, 0));
	PUT(FTRP(ptr), PACK(size, 0));
	coalesce(ptr);
}


/*
 * mm_realloc - Implemented simply in terms of mm_malloc and mm_free
 */
void *mm_realloc(void *ptr, size_t size)
{
    void *oldptr = ptr;
    void *newptr;
	void *nextptr;

	size_t oldsize;
    size_t extendsize;
	size_t asize;	/* 与malloc中的asize相同，表示实际分配的块大小 */
	size_t sizesum;
	
	if(size == 0){			//如果size=0，直接释放ptr
        mm_free(ptr);
        return NULL;
    }
    if(ptr == NULL)			//如果ptr=null，直接分配size
        return mm_malloc(size);

	/* 与malloc相同，使asize满足八字节对齐 */
	oldsize = GET_SIZE(HDRP(ptr));
	asize = ALIGN(size + DSIZE);	//八字节对齐
	

	
	if (asize == oldsize) {						/* 要分配的大小等于块大小，由于块已经满足要求，无需执行操作 */
		return ptr;
	} else if (asize < oldsize) {					/* 要分配的大小小于块大小，直接进行place */
		PUT(HDRP(ptr), PACK( oldsize, 1));
        PUT(FTRP(ptr), PACK( oldsize, 1));
		return ptr;
	} 
	else 
	{										/* 要分配的大小大于块大小， */
		/* 如果下一个块是空块且大小合适，将下一个块与此块合并 */
		nextptr = NEXT_BLKP(ptr);
		sizesum = GET_SIZE(HDRP(nextptr))+oldsize;
		if (!GET_ALLOC(HDRP(nextptr)) && sizesum >= asize) {	
			/* 分配到合并出的新块中 */
			PUT(HDRP(ptr), PACK(sizesum, 0));
			place(ptr, asize);
			return ptr;
		} else {		/* 如果下一个块不满足要求，执行malloc */
			newptr = find_fit(asize);
			if (newptr == NULL) {
				extendsize = MAX(asize, CHUNKSIZE);
				if ((newptr = extend_heap(extendsize/WSIZE)) == NULL) {
					return NULL;
				}
			}
			place(newptr, asize);
			memcpy(newptr, oldptr, oldsize-2*WSIZE);
			mm_free(oldptr);
			return newptr;
		}
	}

}















