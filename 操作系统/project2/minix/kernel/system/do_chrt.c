#include "kernel/system.h"
#include "kernel/vm.h"
#include <signal.h>
#include <string.h>
#include <assert.h>

#include <minix/endpoint.h>
#include <minix/u64.h>



/*===========================================================================*
 *				do_chrt					     *
 *===========================================================================*/
int do_chrt(struct proc * caller, message * m_ptr)
{
  //用消息结构体中的进程号，通过proc_addr定位内核中进程地址，然后将deadline赋值给该进程的p_deadline
  struct proc *rp;
  long deadline;

  deadline=m_ptr->m2_l1;

  rp=proc_addr(m_ptr->m2_i1);
  rp->p_deadline=deadline;
  return OK;
}

