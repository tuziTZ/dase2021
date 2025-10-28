#include <stdio.h>
#include <unistd.h>
#include <pwd.h>
#include <curses.h>
#include <sys/times.h>
#include <stdlib.h>
#include <limits.h>
#include <termcap.h>
#include <termios.h>
#include <time.h>
#include <string.h>
#include <signal.h>
#include <fcntl.h>
#include <errno.h>
#include <dirent.h>
#include <assert.h>

#include <sys/wait.h>
#include <sys/times.h>
#include <sys/types.h>
#include <sys/time.h>
#include <sys/select.h>

#include <minix/com.h>
#include <minix/config.h>
#include <minix/type.h>
#include <minix/endpoint.h>
#include <minix/const.h>
#include <minix/u64.h>
#include <paths.h>
#include <minix/procfs.h>

#define MAXNUM 512
#define MAXLINE 512
#define MAXARGS 128
#define STD_INPUT 0
#define STD_OUTPUT 1

// #define NR_TASKS 4 
// #define KERNEL -3
// #define IDLE -4

#define USED 1
#define IS_TASK 2
#define IS_SYSTEM 4
#define BLOCKED 8
#define PSINFO_VERSION 0

#define STATE_RUN 'R'
const char *cputimenames[] = {"user", "ipc", "kernelcall"};
#define CPUTIMENAMES ((sizeof(cputimenames)) / (sizeof(cputimenames[0]))) //恒等于3
#define CPUTIME(m, i) (m & (1 << (i)))     

char prompt[] = "$ "; 
char workdir[32];
char his[MAXNUM][MAXLINE];
int hiscount=0;
char cmdline[MAXLINE]={0};
int e=0;
int status;
pid_t pid;
char *path = NULL;
unsigned int nr_procs, nr_tasks;
int slot = -1;
int nr_total;

struct proc
{
    int p_flags;
    endpoint_t p_endpoint;           //端点
    pid_t p_pid;                     //进程号
    u64_t p_cpucycles[CPUTIMENAMES]; // CPU周期
    int p_priority;                  //动态优先级
    endpoint_t p_blocked;            //阻塞状态
    time_t p_user_time;              //用户时间
    vir_bytes p_memory;              //内存
    uid_t p_effuid;                  //有效用户ID
    int p_nice;                      //静态优先级
    char p_name[PROC_NAME_LEN + 1];  //名字
};

struct proc *proc = NULL, *prev_proc = NULL;

struct tp
{
    struct proc *p;
    u64_t ticks;
};

void unix_error(char *msg);
void run_child(int count,int fd[20][2],char **argv,int k,int i_list[MAXLINE]);
pid_t Fork(void);
int parseline(char *buf ,char **argv);
void eval(char* cmdline);
//读取/proc/kinfo得到总的进程和任务数num_total
void getkinfo();
//在/proc/meminfo中查看内存信息，计算出内存大小并打印
int print_memory();
//计算总体CPU使用占比并打印结果
void print_procs(struct proc *proc1, struct proc *proc2, int cputimemode);
//计算cputicks
u64_t cputicks(struct proc *p1, struct proc *p2, int timemode);
void get_procs();
//读取目录下每一个文件信息
void parse_dir();
//在/proc/pid/psinfo中，查看进程pid的信息
void parse_file(pid_t pid);

int main() 
{
  while(1){
    
    getcwd(workdir,32);
    printf("%s %s",workdir, prompt);
    fflush(stdout);
    fgets(cmdline,MAXLINE,stdin);
    strcpy(his[hiscount],cmdline);
    hiscount+=1;
    if (feof(stdin)) {
      fflush(stdout);
      exit(0);
    }
    eval(cmdline);
    fflush(stdout);
    if(e==1){
      break;
    }
  }
  return 0;
}

void eval(char* cmdline){


  char buf[MAXLINE];

  char *argv[MAXARGS];

  int argnum;
    
  strcpy(buf, cmdline);
  argnum=parseline(buf,argv);


  //空命令直接返回
  if(argv[0]==NULL||!strcmp(argv[0], "&")){
    return;
  }
  //判断是否为shell内置命令
  //5.cd命令
  if(!strcmp(argv[0],"cd")){
    if(chdir(argv[1])==-1){
      printf("No such file or directory\n");
    };
    return;
  }
  //8.history命令
  if(!strcmp(argv[0],"history")){
    if(argv[1]==NULL){
      printf("You need the second argument\n");
      return;
    }
    int startnum=hiscount-atoi(argv[1]);
    if(startnum<0){
      printf("The number of instructions entered is less than the number of instructions you requested\n");
      startnum=0;
    }
    for(int i=startnum;i<hiscount;i++){
      printf("%s",his[i]);
    }
    return;
  }
  //7.exit命令
  if(!strcmp(argv[0],"exit")){
    e=1;
    return;
  }
  //6.mytop命令
  if (!strcmp(argv[0], "mytop")){
    
    int cputimemode = 1; //计算CPU的时钟周期
    getkinfo();
    print_memory();
    //得到prev_proc
    get_procs();
    if (prev_proc == NULL)
    {
      get_procs(); //得到proc
    }
    print_procs(prev_proc, proc, cputimemode);
    return;
  }

  // if(!strcmp(argv[0],"mytop")){
  //   unsigned int pagesize;
	//   unsigned long total, free, largest, cached;
  //   FILE *f1=fopen("/proc/meminfo","r");
    
  //   fscanf(f1,"%u %lu %lu %lu %lu", &pagesize, &total, &free,
	// 		&largest, &cached);
  //   fclose(f1);
  //   unsigned long total_mem,free_mem,cached_mem;
  //   total_mem=(pagesize*total)/1024;
  //   free_mem=(pagesize*free)/1024;
  //   cached_mem=(pagesize*cached)/1024;

    
    
  //   //先获取进程总数
  //   unsigned nr_procs,nr_tasks;
  //   int nr_total;
  //   f1=fopen("/proc/kinfo","r");
  //   fscanf(f1, "%u %u", &nr_procs, &nr_tasks);
  //   fclose(f1);
  //   nr_total=(int)(nr_procs+nr_tasks);
  //   //计算ticks
  //   u64_t total_ticks = 0;
  //   u64_t runticks = 0;
  //   char path[32];
  //   for(int j=0;j<nr_total;j++){
  //     sprintf(path,"/proc/%d/psinfo",j);
  //     if((f1=fopen(path,"r"))==NULL){
  //       continue;
  //     }
  //     int version,endpt,blocked,priority;
  //     char state,type,name[256];
  //     unsigned long user_time;
  //     fscanf(f1,"%d %c %d %255s %c %d %d %lu",&version,&type,&endpt,
  //     name,&state,&blocked,&priority,&user_time);
  //     total_ticks+=user_time;
  //     if((int)state==82){
  //       runticks+=user_time;
  //     }
      
  //   }

  //   printf("%ld %ld %ld %6.2f%%\n",total_mem,free_mem,cached_mem,100.0 * runticks / total_ticks);

  //   return;
  // }
  // if(!strcmp(argv[0],"mytop")){
  //   char *num_list[32];
  //   int f1=open("/proc/meminfo",O_RDONLY);
  //   read(f1,buf,MAXLINE);
    
  //   parseline(buf,num_list);
  //   int pagesize=atoi(num_list[0]);
  //   int total=atoi(num_list[1]);
  //   int free_=atoi(num_list[2]);
  //   int cached=atoi(num_list[4]);

  //   int total_mem=(pagesize*total)/1024;
  //   int free_mem=(pagesize*free_)/1024;
  //   int cached_mem=(pagesize*cached)/1024;
  //   //计算ticks
  //   //先获取进程总数
  //   f1=open("/proc/kinfo",O_RDONLY);
  //   read(f1,buf,MAXLINE);
  //   parseline(buf,num_list);

  //   int maxpro=atoi(num_list[0])+atoi(num_list[1]);
  //   long a_ticks=0;
  //   long r_ticks=0;
  //   char path[32];
  //   for(int j=0;j<maxpro;j++){
  //     sprintf(path,"/proc/%d/psinfo",j);
  //     if((f1=open(path,O_RDONLY))==-1){
  //       continue;
  //     }
  //     read(f1,buf,MAXLINE);
  //     parseline(buf,num_list);
  //     if(!strcmp(num_list[4],"R")){
  //       r_ticks+=atoi(num_list[7]);
  //     }
  //     a_ticks+=atoi(num_list[7]);
  //   }
  //   printf("%d %d %d %lf%%\n",total_mem,free_mem,cached_mem,(double)(r_ticks)*100.0/(double)a_ticks);

  //   return;
  // }




  //执行program命令
  //3.判别带有管道的命令
  int i=0;
  int i_list[MAXLINE]={0};
  int count=0;

  while(i<argnum){
    if(!strcmp(argv[i],"|")){
      i_list[count]=i+1;
      count+=1;
    }
    i++;
  }


  //2.判别带有重定向内容的命令
  i=0;
  int dupset=0;
  char* filename=NULL;
  while(i<argnum){
    if(!strcmp(argv[i],">")){
      dupset=1;
      filename=argv[i+1];
      i_list[0]=i+1;
      break;
    }
    else if(!strcmp(argv[i],">>")){
      dupset=2;
      filename=argv[i+1];
      i_list[0]=i+1;
      break;
    }
    else if(!strcmp(argv[i],"<")){
      dupset=3;
      filename=argv[i+1];
      i_list[0]=i+1;
      break;
    }
    i++;
  }


  //4.判别是否后台运行
  int bg;
  bg=!strcmp(argv[argnum-1],"&");

  if((dupset!=0&&count!=0)||(dupset!=0&&bg!=0)||(count!=0&&bg!=0)){
    printf("Sorry, the program can't parsing this command\n");
    return;   
  }

  for (int k=0;k<argnum;k++){
    if(i_list[k]!=0){
      argv[i_list[k]-1]='\0';
    }
    else{
      break;
    }
  }

  //执行管道指令
  if (count!=0){
    int k=0;

    int fd[20][2];
    for(i=0;i<count;i++){
      pipe(fd[i]);
    }
    if((pid=Fork())==0){
      if((pid=Fork())==0){
        run_child(count,fd,argv,count-1,i_list);
        exit(0);
      }
      else{
        for(i=0;i<count;i++){
          if(i==count-1){
            close(fd[count-1][1]);
          }
          else{
            close(fd[i][0]);
            close(fd[i][1]);

          }
        }
        dup2(fd[count-1][0],STDIN_FILENO);

        waitpid(pid,&status,0);
        if(execvp(argv[i_list[count-1]],argv+i_list[count-1])<0){
          printf("Command not found\n");
          return;
        }
      }
    }
    else{
      for(i=0;i<count;i++){
        close(fd[i][0]);
        close(fd[i][1]);
      }
      waitpid(pid,&status,0);
      return;
    }
    //第一个进程从stdin读入数据，写入fd[0][1]
    //中间的(第i个)进程从fd[k-1][0]读入数据，写入fd[k][1]
    //最后一个进程从fd[count-1][0]读入数据，写入stdout
    
  }
  //执行重定向到文件
  if(dupset!=0){

    if(dupset==1){
      if((pid=Fork())==0){
        
        // close(STD_OUTPUT);
        int fd=open(filename,O_WRONLY);
        dup2(fd,STD_OUTPUT);
        if(execvp(argv[0],argv)<0){
          printf("Command not found\n");
          return;
        }
      }
      else{
        waitpid(pid,&status,0);
        return;
      }
    }
    else if(dupset==2){
      if((pid=Fork())==0){
        close(STD_OUTPUT);
        int fd=open(filename,O_APPEND);
        dup(fd);
        
        if(execvp(argv[0],argv)<0){
          printf("Command not found\n");
          return;
        }
      }
      else{
        waitpid(pid,&status,0);
        return;
      }
    }
    else{
      if((pid=Fork())==0){
        close(STD_INPUT);
        int fd=open(filename,O_RDONLY);
        dup(fd);
        if(execvp(argv[0],argv)<0){
          printf("Command not found\n");
          return;
        }
      }
      else{
        waitpid(pid,&status,0);
        return;
      }
    }
  }
  // 执行后台运行
  if(bg!=0){
    printf("The program is running background\n");
    argv[argnum-1]="\0";
    if((Fork())==0){
      signal(SIGCHLD,SIG_IGN);
      int fd=open("/dev/null",O_RDONLY);

      dup2(fd,STDIN_FILENO);
      dup2(fd,STDOUT_FILENO);
      dup2(fd,STDERR_FILENO);

      if(execvp(argv[0],argv)<0){
          printf("Command not found\n");
          return;
        }
    }
    else{
      // waitpid(-1,&status,WNOHANG|WUNTRACED);
      return;
    }
  }
  // 执行前台运行

  if((pid=Fork())==0){
    if(execvp(argv[0],argv)<0){
          printf("Command not found\n");
          return;
        }
  }
  else{
    waitpid(pid,&status,0);
    return;
  }
}

void getkinfo()
{
  FILE *fp;
  if ((fp = fopen("/proc/kinfo", "r")) == NULL)
  {
    fprintf(stderr, "opening /proc/kinfo failed\n");
    exit(1);
  }

  if (fscanf(fp, "%u %u", &nr_procs, &nr_tasks) != 2)
  {
    fprintf(stderr, "reading from /proc/kinfo failed");
    exit(1);
  }

  fclose(fp);

  // nr_total是一个全局变量
  nr_total = (int)(nr_procs + nr_tasks);
}

int print_memory()
{
  FILE *fp;
  unsigned long pagesize, total, free, largest, cached;

  if ((fp = fopen("/proc/meminfo", "r")) == NULL)
  {
    return 0;
  }

  if (fscanf(fp, "%lu %lu %lu %lu %lu", &pagesize, &total, &free, &largest, &cached) != 5)
  {
    fclose(fp);
    return 0;
  }

  fclose(fp);

  printf("main memory: %ldk total,%ldk free,%ldk contig free,%ldk cached\n", (pagesize * total) / 1024,
          (pagesize * free) / 1024, (pagesize * largest) / 1024, (pagesize * cached) / 1024);

  return 1;
}

void get_procs()
{
  struct proc *p=NULL;
  int i;
  slot=-1;
  //交换了prev_proc和proc
  p = prev_proc;
  prev_proc = proc;
  proc = p;

  if (proc == NULL)
  {
    // proc是struct proc的集合，申请了nr_total个proc的空间
    proc = malloc(nr_total * sizeof(proc[0])); // struct proc的大小
    if (proc == NULL)
    {
      fprintf(stderr, "Out of memory!\n");
      exit(0);
    }
  }

  for (i = 0; i < nr_total; i++)
  {
    proc[i].p_flags = 0;
  }

  parse_dir();
}

void parse_dir()
{
    DIR *p_dir;
    struct dirent *p_ent;
    pid_t pid;
    char *end; //指向第一个不可转换的字符位置的指针

    if ((p_dir = opendir("/proc")) == NULL)
    {
        perror("opendir on /proc");
        exit(0);
    }

    //读取目录下的每一个文件信息
    for (p_ent = readdir(p_dir); p_ent != NULL; p_ent = readdir(p_dir))
    {
        // long int strtol (const char* str, char** endptr, int base);
        //将字符串转化为长整数，endptr第一个不可转换的位置的字符指针，base要转换的进制
        //合法字符为0x1-0x9
        pid = strtol(p_ent->d_name, &end, 10);
        //由文件名获取进程号
        // pid由文件名转换得来
        // ASCII码对照表，NULL的值为0
        if (!end[0] && pid != 0)
        {
            parse_file(pid);
        }
    }
    closedir(p_dir);
}

// proc/pid/psinfo
void parse_file(pid_t pid)
{
    // PATH_MAX定义在头文件<limits.h>，对路径名长度的限制
    char path[PATH_MAX], name[256], type, state;
    int version, endpt, effuid;         //版本，端点，有效用户ID
    unsigned long cycles_hi, cycles_lo; //高周期，低周期
    FILE *fp;
    struct proc *p;
    int i;
    //将proc/pid/psinfo路径写入path
    sprintf(path, "/proc/%d/psinfo", pid);

    if ((fp = fopen(path, "r")) == NULL)
    {
        return;
    }

    if (fscanf(fp, "%d", &version) != 1)
    {
        fclose(fp);
        return;
    }

    if (version != PSINFO_VERSION)
    {
        fputs("procfs version mismatch!\n", stderr);
        exit(1);
    }

    if (fscanf(fp, " %c %d", &type, &endpt) != 2)
    {
        fclose(fp);
        return;
    }

    slot++; //顺序取出每个proc让所有task的slot不冲突

    if (slot < 0 || slot >= nr_total)
    {
        fprintf(stderr, "mytop:unreasonable endpoint number %d\n", endpt);
        fclose(fp);
        return;
    }

    p = &proc[slot]; //取得对应的struct proc

    if (type == TYPE_TASK)
    {
        p->p_flags |= IS_TASK; // 0x2 倒数第二位标记为1
    }
    else if (type == TYPE_SYSTEM)
    {
        p->p_flags |= IS_SYSTEM; // 0x4 倒数第三位标记为1
    }
    
    p->p_endpoint = endpt;
    p->p_pid = pid;
    //%*u添加了*后表示文本读入后不赋给任何变量
    if (fscanf(fp, " %255s %c %d %d %lu %*u %lu %lu",
               name, &state, &p->p_blocked, &p->p_priority,
               &p->p_user_time, &cycles_hi, &cycles_lo) != 7)
    {
        fclose(fp);
        return;
    }

    // char*strncpy(char*dest,char*src,size_tn);
    //复制src字符串到dest中，大小由tn决定
    strncpy(p->p_name, name, sizeof(p->p_name) - 1);
    p->p_name[sizeof(p->p_name) - 1] = 0;

    if (state != STATE_RUN)
    {
        p->p_flags |= BLOCKED; // 0x8 倒数第四位标记为1
    }

    // user的CPU周期
    p->p_cpucycles[0] = make64(cycles_lo, cycles_hi);
    p->p_flags |= USED; //最低位标记位1

    fclose(fp);
}

u64_t cputicks(struct proc *p1, struct proc *p2, int timemode)
{
    int i;
    u64_t t = 0;
    for (i = 0; i < CPUTIMENAMES; i++)
    {
        if (!CPUTIME(timemode, i))
        {
            continue;
        }
        // timemode==1只有i等于0时CPUTIME才等于1
        //只有i=0时会执行后面的，即只计算了CPU的时钟周期不会对另外两个做计算
        // p_cpucycles第二个值为ipc，第三个值为kernelcall的数量
        //如果两个进程相等则作差求时间差
        if (p1->p_endpoint == p2->p_endpoint)
        {
            t = t + p2->p_cpucycles[i] - p1->p_cpucycles[i];
        }
        else
        { //否则t直接加上p2
            t = t + p2->p_cpucycles[i];
        }
    }
    return t;
}

void print_procs(struct proc *proc1, struct proc *proc2, int cputimemode)
{
    int p, nprocs;
    u64_t systemticks = 0;
    u64_t userticks = 0;
    u64_t total_ticks = 0;
    u64_t idleticks = 0;
    u64_t kernelticks = 0;
    int blockedseen = 0;
    //创建了一个struct tp的结构体数组
    static struct tp *tick_procs = NULL;
    
    if (tick_procs == NULL)
    {
        tick_procs = malloc(nr_total * sizeof(tick_procs[0]));
        if (tick_procs == NULL)
        {
            fprintf(stderr, "Out of memory!\n");
            exit(1);
        }
    }
    
			
    for (p = nprocs = 0; p < nr_total; p++)
    {
      // if(!(proc2[p].p_flags & USED)){
      //   continue;
      // }
      // printf("p:%d\n",p);
      u64_t uticks;
      // printf("p=%d ",p);
      tick_procs[nprocs].p = proc2 + p; //初始化
      // tickprocs的第np个结构体的struct proc *p
      //为proc2第p个文件的struct proc
      // printf("prev_proc pid=%d, proc pid=%d, p=%d\n",proc1[p].p_pid,proc2[p].p_pid,p);
      tick_procs[nprocs].ticks = cputicks(&proc1[p], &proc2[p], cputimemode);
      uticks = cputicks(&proc1[p], &proc2[p], 1);
      total_ticks = total_ticks + uticks;

      
        printf("pid=%d p=%d uticks=%llu\n",tick_procs[nprocs].p->p_pid,p,uticks);
      

      if (tick_procs[nprocs].p->p_pid==-4)
      {
        idleticks = uticks;
        continue;
      }
      if (tick_procs[nprocs].p->p_pid==-1)
      {
        kernelticks = uticks;
      }
      if (!(proc2[p].p_flags & IS_TASK))
      {
        //如果是进程，则看是system还是user
        if (proc2[p].p_flags & IS_SYSTEM)
        {
            systemticks = systemticks + tick_procs[nprocs].ticks;
        }
        else
        {
            userticks = userticks + tick_procs[nprocs].ticks;
        }
      }
      nprocs++;
    }
    if (total_ticks == 0)
    {
        return;
    }
    printf("CPU states: %6.2f%% user, ", 100.0 * userticks / total_ticks);
    printf("%6.2f%% system, ", 100.0 * systemticks / total_ticks);
    printf("%6.2f%% kernel, ", 100.0 * kernelticks / total_ticks);
    printf("%6.2f%% idle\n", 100.0 * idleticks / total_ticks);
}


int parseline(char *cmdline, char** argv){
  static char array[MAXLINE];
  char* buf=array;
  char* delim;
  int bg=0;
  int argnum;
  strcpy(buf,cmdline);
  buf[strlen(buf)-1]=' ';
  while(*buf && (*buf==' ')){
    buf++;
  }
  argnum=0;
  delim=strchr(buf,' ');
  while(delim){

    argv[argnum++]=buf;
    *delim='\0';
    buf=delim+1;
    while(*buf && (*buf==' ')){
      buf++;
    }
    delim=strchr(buf,' ');

  }
  argv[argnum]=NULL;

  return argnum;

}
pid_t Fork(void){
  pid_t pid;

  if ((pid = fork()) < 0)
    unix_error("Fork error");
  return pid;
}


void unix_error(char *msg){
    fprintf(stdout, "%s: %s\n", msg, strerror(errno));
    exit(1);
}

void run_child(int count,int fd[20][2],char **argv,int k,int i_list[MAXLINE]){
  
  if((pid=Fork())==0){
    if(k==0){
      for(int i=0;i<count;i++){
        if(i==0){
          close(fd[0][0]);

        }
        else{
          close(fd[i][0]);
          close(fd[i][1]);
        }
      }
      dup2(fd[0][1],STD_OUTPUT);
      if(execvp(argv[0],argv)<0){
          printf("Command not found\n");
          return;
        }
    }
    else{
      if((pid=Fork())==0){
        run_child(count,fd,argv,k-1,i_list);
      }
      else{
        for(int i=0;i<count;i++){
          if(i==k){
            close(fd[k][0]);
          }
          else if(i==k-1){
            close(fd[k-1][1]);

          }
          else{
            close(fd[i][0]);
            close(fd[i][1]);

          }
        }
        waitpid(pid,&status,0);
        dup2(fd[k-1][0],STDIN_FILENO);
        dup2(fd[k][1],STDOUT_FILENO);
        if(execvp(argv[i_list[k-1]],argv+i_list[k-1])<0){
          printf("Command not found\n");
          return;
        }
        
      }
    }
  }
  else{
    for(int i=0;i<count;i++){
        close(fd[i][0]);
        close(fd[i][1]);
      }
    waitpid(pid,&status,0);
    return;
  }
}