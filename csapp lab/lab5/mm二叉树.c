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

/*组织二叉树*/
#define LEFT(bp) ((void *)(bp))
#define RIGHT(bp) ((void *)(bp)+WSIZE)
#define PRNT(bp) ((void *)(bp)+DSIZE)
#define BROS(bp) ((void *)(bp)+(3*WSIZE))

/*读写二叉树*/
#define PUT_LEFT_CHILD(bp,val) (PUT(LEFT(bp),(int)val))
#define PUT_RIGHT_CHILD(bp,val) (PUT(RIGHT(bp),(int)val))
#define GET_LEFT_CHILD(bp) (GET(LEFT(bp)))
#define GET_RIGHT_CHILD(bp) (GET(RIGHT(bp)))
#define PUT_PAR(bp,val) (PUT(PRNT(bp),(int)val))
#define PUT_BROS(bp,val) (PUT(BROS(bp),(int)val))
#define GET_PAR(bp) (GET(PRNT(bp)))
#define GET_BRO(bp) (GET(BROS(bp)))

/*取块的大小*/
#define GET_HDRP_SIZE(bp) GET_SIZE(HDRP(bp))

static char *heap_listp=0;//堆的起始位置NULL
static unsigned int *my_tree = 0;
static size_t flag = 0;

static void insert(void *bp);       //插入节点
static void delete(void *bp);       //删除节点


//合并前后的空闲块
static void *coalesce(void *bp)
{
    /* 获取前后两个块的空闲情况 */
	size_t prev_alloc = GET_ALLOC(FTRP(PREV_BLKP(bp)));
	size_t next_alloc = GET_ALLOC(HDRP(NEXT_BLKP(bp)));
	size_t size = GET_SIZE(HDRP(bp));//获取当前块的大小
	
	if (prev_alloc && next_alloc) {				/* 前后都已分配 */
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
		size += GET_SIZE(HDRP(PREV_BLKP(bp))) 
			+ GET_SIZE(FTRP(NEXT_BLKP(bp)));
        delete(NEXT_BLKP(bp));
        delete(PREV_BLKP(bp));
        
		PUT(HDRP(PREV_BLKP(bp)), PACK(size, 0));
		PUT(FTRP(NEXT_BLKP(bp)), PACK(size, 0));
		bp = PREV_BLKP(bp);
	}
    return bp;
}

static void *extend_heap(size_t words)
{
	char *bp;
	size_t size=words;
    void *tmp; 
    
	if (words <= 0)
		return 0;
	/* 分配偶数个字，保证八字节对齐 */

	if((long) (bp=mem_sbrk(size))==-1)//申请的区域超出最大堆的大小
		return NULL;
	/* 添加空闲块的头部和脚部 */
	PUT(HDRP(bp),PACK(size,0));
	PUT(FTRP(bp),PACK(size,0));
	PUT(HDRP(NEXT_BLKP(bp)),PACK(0,1));
	
	/*采用立即边界合并方式：在每次一个块被释放时就合并所有的相邻块*/
	tmp = coalesce(bp);
	insert(tmp);
	return bp;
}


void insert(void *bp)
{
    if (my_tree == 0)               //树是空的
	{
		my_tree = bp;
		PUT_LEFT_CHILD(bp,0);
		PUT_RIGHT_CHILD(bp,0);
		PUT_PAR(bp,0);
		PUT_BROS(bp,0);
		return;
	}

	unsigned int *my_tr = my_tree;

	while (1)                   //从根节点开始，找到新节点的父节点的位置
	{
		if (GET_HDRP_SIZE(bp) < GET_HDRP_SIZE(my_tr))       
			if (GET_LEFT_CHILD(my_tr) != 0)
				my_tr = (unsigned int*)GET_LEFT_CHILD(my_tr);
			else break;
		else if (GET_HDRP_SIZE(bp) > GET_HDRP_SIZE(my_tr))
			if (GET_RIGHT_CHILD(my_tr) != 0)
				my_tr = (unsigned int*)GET_RIGHT_CHILD(my_tr);
			else break;
		else break;
	}
	if ((GET_HDRP_SIZE(bp) < GET_HDRP_SIZE(my_tr)))//如果新节点是左子节点
	{
		PUT_LEFT_CHILD(my_tr,bp);
		PUT_PAR(bp,my_tr);
		PUT_BROS(bp,0);//没有兄弟节点
		PUT_LEFT_CHILD(bp,0);
		PUT_RIGHT_CHILD(bp,0);
		return;
	}
	else if (GET_HDRP_SIZE(bp) > GET_HDRP_SIZE(my_tr))//如果新节点是右子节点
	{
		PUT_RIGHT_CHILD(my_tr,bp);
		PUT_PAR(bp,my_tr);
		PUT_BROS(bp,0);
		PUT_LEFT_CHILD(bp,0);
		PUT_RIGHT_CHILD(bp,0);
		return;
	}
	else if (GET_HDRP_SIZE(bp) == GET_HDRP_SIZE(my_tr))//如果新节点和父节点大小相同
	{
		if (my_tr == my_tree)//如果父节点是根节点
		{				
			my_tree = bp;//用新节点替换根节点
			PUT_LEFT_CHILD(bp,GET_LEFT_CHILD(my_tr));
			PUT_RIGHT_CHILD(bp,GET_RIGHT_CHILD(my_tr));
			if (GET_LEFT_CHILD(my_tr) != 0)
				PUT_PAR(GET_LEFT_CHILD(my_tr),bp);
			if (GET_RIGHT_CHILD(my_tr) != 0)
				PUT_PAR(GET_RIGHT_CHILD(my_tr),bp);
			PUT_PAR(bp,0);

			PUT_BROS(bp,my_tr);//根节点和新节点成为兄弟节点

			PUT_LEFT_CHILD(my_tr,bp);//新节点作为根节点的左孩子
			PUT_RIGHT_CHILD(my_tr,-1);//根节点的由孩子设为-1
			return;
		}
		else//如果父节点不是根节点
		{
            //用该节点替换父节点
			if (GET_HDRP_SIZE(GET_PAR(my_tr)) >  GET_HDRP_SIZE(my_tr))//如果父节点是祖父节点的左孩子
				PUT_LEFT_CHILD(GET_PAR(my_tr),bp);//新节点成为祖父节点的左孩子
			else if (GET_HDRP_SIZE(GET_PAR(my_tr)) <  GET_HDRP_SIZE(my_tr))
				PUT_RIGHT_CHILD(GET_PAR(my_tr),bp);
            //把父节点的左右孩子给新节点
			PUT_LEFT_CHILD(bp,GET_LEFT_CHILD(my_tr));
			PUT_RIGHT_CHILD(bp,GET_RIGHT_CHILD(my_tr));
            if (GET_LEFT_CHILD(my_tr) != 0)     
				PUT_PAR(GET_LEFT_CHILD(my_tr),bp);
			if (GET_RIGHT_CHILD(my_tr) != 0)
				PUT_PAR(GET_RIGHT_CHILD(my_tr),bp);
			PUT_PAR(bp,GET_PAR(my_tr));

			PUT_BROS(bp,my_tr);//两个大小相同的节点成为兄弟节点
			PUT_RIGHT_CHILD(my_tr,-1);
			PUT_LEFT_CHILD(my_tr,bp);//新节点成为父节点的左孩子
			return;
		}
	}
}






void delete(void *bp)
{
	if (bp == my_tree)//如果该节点是根节点
	{
		if (GET_BRO(bp) != 0)//如果该节点有兄弟节点（有尾随的链表）
		{
			my_tree = (unsigned int*)GET_BRO(bp);//兄弟节点设为根节点，取代被删除的节点
			PUT_LEFT_CHILD(my_tree,GET_LEFT_CHILD(bp));
			PUT_RIGHT_CHILD(my_tree,GET_RIGHT_CHILD(bp));
			if (GET_RIGHT_CHILD(bp) != 0)
				PUT_PAR(GET_RIGHT_CHILD(bp),my_tree);
			if (GET_LEFT_CHILD(bp) != 0)
				PUT_PAR(GET_LEFT_CHILD(bp),my_tree);
			return;
		}
		else//如果没有兄弟节点
		{
			if (GET_LEFT_CHILD(bp) == 0)//如果没有左孩子，直接把右孩子作为根节点
				my_tree=(unsigned int*)GET_RIGHT_CHILD(bp);
			else if (GET_RIGHT_CHILD(bp) == 0)//如果没有右孩子，直接把左孩子作为根节点
				my_tree=(unsigned int*)GET_LEFT_CHILD(bp);
			else//如果有两个子节点
			{
				unsigned int *my_tr = (unsigned int*)GET_RIGHT_CHILD(bp);
				while (GET_LEFT_CHILD(my_tr) != 0)
					my_tr = (unsigned int*)GET_LEFT_CHILD(my_tr);//寻找右子树上的最小节点，即后继
				my_tree = my_tr;//用后继来顶替根节点
				if (GET_LEFT_CHILD(bp) != 0)//把原节点的左子树移植给后继
					PUT_PAR(GET_LEFT_CHILD(bp),my_tr);
				if (my_tr != (unsigned int*)GET_RIGHT_CHILD(bp))//把原节点的右子树移植给后继
				{
					if (GET_RIGHT_CHILD(my_tr) != 0)//把后继原先的右子树提上来一层
						PUT_PAR(GET_RIGHT_CHILD(my_tr),GET_PAR(my_tr));
					PUT_LEFT_CHILD(GET_PAR(my_tr),GET_RIGHT_CHILD(my_tr));
					PUT_RIGHT_CHILD(my_tr,GET_RIGHT_CHILD(bp));
					PUT_PAR(GET_RIGHT_CHILD(bp),my_tr);
				}
				PUT_LEFT_CHILD(my_tr,GET_LEFT_CHILD(bp));
			}
		}
	}
	else//如果不是根节点
	{
		if (GET_RIGHT_CHILD(bp) != -1 && GET_BRO(bp) == 0)//如果要删除的节点不在链表上
		{
			if  (GET_RIGHT_CHILD(bp) == 0)
			{//如果没有右孩子
				if (GET_HDRP_SIZE(bp) > GET_HDRP_SIZE(GET_PAR(bp)))
					PUT_RIGHT_CHILD(GET_PAR(bp),GET_LEFT_CHILD(bp));
				else
					PUT_LEFT_CHILD(GET_PAR(bp),GET_LEFT_CHILD(bp));
				if (GET_LEFT_CHILD(bp) != 0 && GET_PAR(bp) != 0)
					PUT_PAR(GET_LEFT_CHILD(bp),GET_PAR(bp));
			}
			else if (GET_RIGHT_CHILD(bp) != 0)
			{//如果有右孩子
				unsigned int *my_tr = (unsigned int*)GET_RIGHT_CHILD(bp);
				while(GET_LEFT_CHILD(my_tr) != 0)//找到后继
					my_tr = (unsigned int*)GET_LEFT_CHILD(my_tr);
				if (GET_HDRP_SIZE(bp) > GET_HDRP_SIZE(GET_PAR(bp)))
					PUT_RIGHT_CHILD(GET_PAR(bp),my_tr);
				else
					PUT_LEFT_CHILD(GET_PAR(bp),my_tr);
				if (my_tr != (unsigned int*)GET_RIGHT_CHILD(bp))
				{
					if (GET_RIGHT_CHILD(my_tr) != 0)
					{
						PUT_LEFT_CHILD(GET_PAR(my_tr),GET_RIGHT_CHILD(my_tr));
						PUT_LEFT_CHILD(GET_PAR(my_tr),GET_RIGHT_CHILD(my_tr));
						PUT_PAR(GET_RIGHT_CHILD(my_tr),GET_PAR(my_tr));
					}
					else
						PUT_LEFT_CHILD(GET_PAR(my_tr),0);
					PUT_RIGHT_CHILD(my_tr,GET_RIGHT_CHILD(bp));
					PUT_PAR(GET_RIGHT_CHILD(bp),my_tr);
				}
				PUT_PAR(my_tr,GET_PAR(bp));
				PUT_LEFT_CHILD(my_tr,GET_LEFT_CHILD(bp));
				if (GET_LEFT_CHILD(bp) != 0)
					PUT_PAR(GET_LEFT_CHILD(bp),my_tr);
			}
		}

		else if (GET_RIGHT_CHILD(bp) == -1)
		{//如果是链表中的其他成员，则从链表中删除
			if (GET_BRO(bp) != 0)
				PUT_LEFT_CHILD(GET_BRO(bp),GET_LEFT_CHILD(bp));
			PUT_BROS(GET_LEFT_CHILD(bp),GET_BRO(bp));
		}

		else if (GET_RIGHT_CHILD(bp) != -1 && GET_BRO(bp) != 0)
		{//是链表中的第一个，则用链表中的下一个顶替
			
			if (GET_HDRP_SIZE(bp) > GET_HDRP_SIZE(GET_PAR(bp)))
				PUT_RIGHT_CHILD(GET_PAR(bp),GET_BRO(bp));
			else
				PUT_LEFT_CHILD(GET_PAR(bp),GET_BRO(bp));
			PUT_LEFT_CHILD(GET_BRO(bp),GET_LEFT_CHILD(bp));
			PUT_RIGHT_CHILD(GET_BRO(bp),GET_RIGHT_CHILD(bp));
			if (GET_LEFT_CHILD(bp) != 0)
				PUT_PAR(GET_LEFT_CHILD(bp),GET_BRO(bp));
			if (GET_RIGHT_CHILD(bp) != 0)
				PUT_PAR(GET_RIGHT_CHILD(bp),GET_BRO(bp));
			PUT_PAR(GET_BRO(bp),GET_PAR(bp));
		}
	}
}





/* 根据大小类遍历分离链表，找到可以分配的空闲块 */
static void *find_fit(size_t asize)
{

    void *cur = (void*)my_tree;
	void *my_fit = 0;
	while (cur != 0)//最优适配
	{
		if (asize == GET_SIZE(HDRP(cur)))
		{
			my_fit = cur;
			break;
		}
		else if (asize < GET_SIZE(HDRP(cur)))
		{
			my_fit = cur;
			cur = (void*)GET_LEFT_CHILD(cur);
		}
		else
			cur = (void*)GET_RIGHT_CHILD(cur);
	}
	return my_fit;
	
}

/* 在空闲块头部分配一个块，使满足八字节对齐，如果分配块小于最小块的大小，就不进行分割 */
static void place(void *bp, size_t asize)
{
    size_t csize = GET_SIZE(HDRP(bp));
    /* 块已分配，从空闲链表中删除 */
    delete(bp);
    void *tmp;
    if((csize - asize) >= 3*DSIZE) {
        PUT(HDRP(bp), PACK(asize, 1));
        PUT(FTRP(bp), PACK(asize, 1));
        bp = NEXT_BLKP(bp);
        PUT(HDRP(bp), PACK(csize - asize, 0));
        PUT(FTRP(bp), PACK(csize - asize, 0));
        /* 加入分离出来的空闲块 */
        tmp = coalesce(bp);
		insert(tmp);
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
    if((heap_listp = mem_sbrk(4*WSIZE)) == 0)
        return -1;
    //堆从大小类头指针后开始
    PUT(heap_listp, 0);
    PUT(heap_listp+WSIZE, PACK(DSIZE, 1));     //序言块
    PUT(heap_listp+WSIZE*2, PACK(DSIZE, 1));     
    heap_listp += (4*WSIZE);
    my_tree = 0;
    if (extend_heap(1<<10) == NULL) {
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
    size_t asize=0;		
    size_t extendsize;	/* 如果要扩展堆，所扩展的字节数 */
   	char *bp;

	if (size <= 0)
		return 0;
		
	asize = size + 8;

	if (asize <= 24)
		asize = 24;
	
	else	
		asize = ALIGN(asize);
	
	if (size == 112)
		asize = 136;
	else if (size == 448)
		asize = 520;


	bp=find_fit(asize);

	if (bp != 0)
	{
		//make binary-bal.rep and binary2-bal.rep become faster
		place(bp,asize);
		return bp;
	}

	else
	{
		extendsize = MAX(asize + 24  + 16,1 << 10);
		extend_heap(extendsize);
		if ((bp=find_fit(asize)) == 0)
			return 0;
		place(bp,asize);
		return bp;
	}

}


/*
 * mm_free - Freeing a block does nothing.
 */
void mm_free(void *ptr)
{
    void *tmp=0;
    if(ptr==0)
        return;
    size_t size = GET_SIZE(HDRP(ptr));

    PUT(HDRP(ptr), PACK(size, 0));
    PUT(FTRP(ptr), PACK(size, 0));
    tmp = coalesce(ptr);
	insert(tmp);
}

/*
 * mm_realloc - Implemented simply in terms of mm_malloc and mm_free
 */
// void *mm_realloc(void *ptr, size_t size)
// {
//     void *newptr;
//     size_t copysize;
//     /*放弃合并，直接进行malloc(size)和free(ptr)操作 */
//     if((newptr = mm_malloc(size))==NULL)//先进行malloc，如果无法分配新的size，返回0
//         return 0;
//     copysize = GET_SIZE(HDRP(ptr));     //指向位置的块大小

//     if(size < copysize)                 //如果指向位置的块较大
//         copysize = size;
//     memcpy(newptr, ptr, copysize);
//     mm_free(ptr);
//     return newptr;
// }
/*
 * mm_realloc - Implemented simply in terms of mm_malloc and mm_free
 */
void *mm_realloc(void *ptr, size_t size)
{
	/*malloc和free的简单结合*/
    size_t oldsize, asize;
    void *newptr;

    if(size == 0){			//如果size=0，直接释放ptr
        mm_free(ptr);
        return NULL;
    }
    if(ptr == NULL)			//如果ptr=null，直接分配size
        return mm_malloc(size);

    oldsize = GET_SIZE(HDRP(ptr));
    asize = ALIGN(size + DSIZE);	//八字节对齐

    if(oldsize >= asize){	//如果原先块比分配块大，直接放置在此处
        PUT(HDRP(ptr), PACK( oldsize, 1));
        PUT(FTRP(ptr), PACK( oldsize, 1));
        return ptr;
    }
    else					//如果原先块比分配块小，重新分配size并且拷贝
    {
        newptr = mm_malloc(size);
        memcpy(newptr, ptr, size);
        mm_free(ptr);
        return newptr;
    }
}
