#include <sys/cdefs.h>
#include "namespace.h"
#include <lib.h>

#include <string.h>
#include <unistd.h>
#include <time.h>

int chrt(long deadline){
  struct timeval tv;
  message m;
  memset(&m, 0, sizeof(m));
  if(deadline<0){//deadline<0 防止溢出，参数值错误，返回0表示调用失败
    return 0;
  }
  
  if(deadline!=0){//如果deadline=0就直接赋为0，作为普通进程
    alarm((unsigned int)deadline);//unistd.h 89行 使进程到期退出
    gettimeofday(&tv,NULL);//包括tv_sec,tv_usec，表示从1970-1-1 00:00到当前的秒数和微秒数
    deadline=tv.tv_sec+deadline;//调用时的时间加上deadline，获取到期时的时间，用于比较进程到期的早晚
  }
  
  m.m2_l1 = deadline;//书p98；文件ipc.h 2026行message结构体的定义、mess_2的定义、访问信息体中内容的宏定义
  return(_syscall(PM_PROC_NR, PM_CHRT, &m));//参考fork.c
}