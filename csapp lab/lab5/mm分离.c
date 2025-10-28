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
#define CHUNKSIZE (336) //初始空闲块的大小和扩展堆时的默认大小
#define CLASS_SIZE 50
#define MINCLASSSIZE 2

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

/*当bp指向一个块时，获得前驱和后继块指针*/
#define GET_PRE(bp) ((unsigned int *)(GET(bp)))
#define GET_SUC(bp) ((unsigned int *)(GET((unsigned int *)bp+1)))

/*获得第num个大小类的头指针指向的地址*/
#define GET_HEAD(num) ((unsigned int *)(GET(heap_listp + WSIZE * num)))

static char *heap_listp=0;//堆的起始位置NULL

static void insert(void *bp);       //插入节点
static void delete(void *bp);       //删除节点
static int search(size_t size);     //查找大小类

//合并前后的空闲块
static void *coalesce(void *bp)
{
    /* 获取前后两个块的空闲情况 */
	size_t prev_alloc = GET_ALLOC(FTRP(PREV_BLKP(bp)));
	size_t next_alloc = GET_ALLOC(HDRP(NEXT_BLKP(bp)));
	size_t size = GET_SIZE(HDRP(bp));//获取当前块的大小
	
	if (prev_alloc && next_alloc) {				/* 前后都已分配 */
		insert(bp);
        return bp;
	} else if (prev_alloc && !next_alloc) {		/* 后一个块空闲 */
		delete(NEXT_BLKP(bp));
        size += GET_SIZE(HDRP(NEXT_BLKP(bp)));
		PUT(HDRP(bp), PACK(size, 0));
		PUT(FTRP(bp), PACK(size, 0));
	} else if (!prev_alloc && next_alloc) {		/* 前一个块空闲 */
		delete(PREV_BLKP(bp));
        size += GET_SIZE(FTRP(PREV_BLKP(bp)));
		PUT(FTRP(bp), PACK(size, 0));
		PUT(HDRP(PREV_BLKP(bp)), PACK(size, 0));
		bp = PREV_BLKP(bp);
	} else if (!prev_alloc && !next_alloc) {	/* 前后都空闲 */
		delete(NEXT_BLKP(bp));
        delete(PREV_BLKP(bp));
        size += GET_SIZE(HDRP(PREV_BLKP(bp))) 
			+ GET_SIZE(FTRP(NEXT_BLKP(bp)));
		PUT(HDRP(PREV_BLKP(bp)), PACK(size, 0));
		PUT(FTRP(NEXT_BLKP(bp)), PACK(size, 0));
		bp = PREV_BLKP(bp);
	}
	insert(bp);
    return bp;
}

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


//头插法
void insert(void *bp)
{
    size_t size = GET_SIZE(HDRP(bp));
    /* 根据块大小找到大小类头结点位置 */
    int num = search(size);
    /* 大小类是空的，直接放 */
    if(GET_HEAD(num) == NULL){
        PUT(heap_listp + WSIZE * num, (int)bp);
        PUT(bp, 0);
        PUT((unsigned int *)bp + 1, 0);
	} else {
        /* bp的后继放头指向的节点 */
		PUT((unsigned int *)bp + 1, (int)GET_HEAD(num));
		/* 头指向节点的前驱放bp */
        PUT(GET_HEAD(num), (int)bp);
        /* bp的前驱为空 */  	
		PUT(bp, 0);
        /* 头指向bp */
		PUT(heap_listp + WSIZE * num, (int)bp);
	}
}


void delete(void *bp)
{
    size_t size = GET_SIZE(HDRP(bp));
    /* 根据块大小找到大小类头结点位置 */
    int num = search(size);
    /* 该节点没有前驱和后继*/
	if (GET_PRE(bp) == NULL && GET_SUC(bp) == NULL) { 
		PUT(heap_listp + WSIZE * num, 0);
	} 
    /* 该节点没有后继*/
    else if (GET_PRE(bp) != NULL && GET_SUC(bp) == NULL) {
		PUT(GET_PRE(bp) + 1, 0);
	} 
    /* 该节点没有前驱*/
    else if (GET_SUC(bp) != NULL && GET_PRE(bp) == NULL){
		PUT(heap_listp + WSIZE * num, (int)GET_SUC(bp));
		PUT(GET_SUC(bp), 0);
	}
    /* 有前驱和后继*/
    else if (GET_SUC(bp) != NULL && GET_PRE(bp) != NULL) {
		PUT(GET_PRE(bp) + 1, (int)GET_SUC(bp));
		PUT(GET_SUC(bp), (int)GET_PRE(bp));
	}
}


int search(size_t size)
{
    int i;
    for(i = MINCLASSSIZE; i <=CLASS_SIZE+2; i++){
        if(size <= (1 << i))
            return i-MINCLASSSIZE;
    }
    return i-MINCLASSSIZE;
}



/* 根据大小类遍历分离链表，找到可以分配的空闲块 */
static void *find_fit(size_t asize)
{
    int num = search(asize);
    unsigned int* bp;
    	/* 如果找不到合适的块，那么就搜索下一个更大的大小类 */
    	while(num < CLASS_SIZE) {
        	bp = GET_HEAD(num);
        	/* 不为空则寻找 */
        	while(bp) {
            	if(GET_SIZE(HDRP(bp)) >= asize){
                	return (void *)bp;
            	}
            	/* 用后继找下一块 */
            	bp = GET_SUC(bp);
        	}
        	/* 找不到则进入下一个大小类 */
        	num++;
		}
    return NULL;
	
}

/* 在空闲块头部分配一个块，使满足八字节对齐，如果分配块小于最小块的大小，就不进行分割 */
static void place(void *bp, size_t asize)
{
    size_t csize = GET_SIZE(HDRP(bp));
    /* 块已分配，从空闲链表中删除 */
    delete(bp);
    if((csize - asize) >= 2*DSIZE) {
        PUT(HDRP(bp), PACK(asize, 1));
        PUT(FTRP(bp), PACK(asize, 1));
        bp = NEXT_BLKP(bp);
        PUT(HDRP(bp), PACK(csize - asize, 0));
        PUT(FTRP(bp), PACK(csize - asize, 0));
        /* 加入分离出来的空闲块 */
        insert(bp);
    }
    else{
        PUT(HDRP(bp), PACK(csize, 1));
        PUT(FTRP(bp), PACK(csize, 1));
    }
}


/* 
 * mm_init - initialize the malloc package.
 */
int mm_init(void)
{
    /* 创建一个最小块，包含序言，头部，脚部，结尾块 */
    if((heap_listp = mem_sbrk((4+CLASS_SIZE)*WSIZE)) == (void *)-1)
        return -1;
    //初始化CLASS_SIZE个大小类头指针
    for(int i = 0; i < CLASS_SIZE; i++){
        PUT(heap_listp + i*WSIZE, 0);
    }
    //堆从大小类头指针后开始
    PUT(heap_listp + CLASS_SIZE * WSIZE, 0);
    PUT(heap_listp + ((1 + CLASS_SIZE)*WSIZE), PACK(DSIZE, 1));     //序言块
    PUT(heap_listp + ((2 + CLASS_SIZE)*WSIZE), PACK(DSIZE, 1));     
    PUT(heap_listp + ((3 + CLASS_SIZE)*WSIZE), PACK(0, 1));         //结尾块

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
    void *newptr;
    size_t copysize;
    /*放弃合并，直接进行malloc(size)和free(ptr)操作 */
    if((newptr = mm_malloc(size))==NULL)//先进行malloc，如果无法分配新的size，返回0
        return 0;
    copysize = GET_SIZE(HDRP(ptr));     //指向位置的块大小

    if(size < copysize)                 //如果指向位置的块较大
        copysize = size;
    memcpy(newptr, ptr, copysize);
    mm_free(ptr);
    return newptr;
}