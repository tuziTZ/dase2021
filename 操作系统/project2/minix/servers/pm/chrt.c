#include "pm.h"
#include <minix/com.h>
#include <minix/callnr.h>
#include "mproc.h"
int do_chrt()
{
  sys_chrt(who_p, m_in);//glo.h 16、17行 who_p,消息结构体
  return (OK);
}