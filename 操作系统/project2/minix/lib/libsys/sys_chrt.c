#include "syslib.h"

int sys_chrt(endpt,m1)
endpoint_t endpt;
message m1;
{
  message m;
  int r;

  m.m2_i1=endpt;//将进程号和deadline放入消息结构体，通过kernelcall传递到内核层
  m.m2_l1=m1.m2_l1;
  r = _kernel_call(SYS_CHRT, &m);
  return r;
}