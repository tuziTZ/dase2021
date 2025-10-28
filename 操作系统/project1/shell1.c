#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/types.h>
#include <fcntl.h>
#include <errno.h>
#include <signal.h>
#include<sys/times.h>
#include <sys/wait.h>

#define MAXNUM 512
#define MAXLINE 512
#define MAXARGS 128
#define STD_INPUT 0
#define STD_OUTPUT 1
// #define SIGCHLD 17

char prompt[] = "$ "; 
char workdir[32];
char his[MAXNUM][MAXLINE];
int hiscount=0;
char cmdline[MAXLINE]={0};
int e=0;
int status;

void unix_error(char *msg);
void Execve(const char *filename, char *const argv[], char *const environ[]);
pid_t Fork(void);
int parseline(char *buf ,char **argv);
void eval(char* cmdline);

int main(int argc, char **argv) 
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
      printf("No such file or directory");
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
  if(!strcmp(argv[0],"mytop")){

    char *num_list[32];
    int f1=open("/proc/meminfo",O_RDONLY);
    read(f1,buf,MAXLINE);
    
    parseline(buf,num_list);
    int pagesize=atoi(num_list[0]);


    int total=atoi(num_list[1]);
    int free_=atoi(num_list[2]);
    int cached=atoi(num_list[4]);

    int total_mem=(pagesize*total)/1024;
    int free_mem=(pagesize*free_)/1024;
    int cached_mem=(pagesize*cached)/1024;
    //计算ticks
    //先获取进程总数
    f1=open("/proc/kinfo",O_RDONLY);
    read(f1,buf,MAXLINE);
    parseline(buf,num_list);

    int maxpro=atoi(num_list[0]);
    long a_ticks=0;
    long r_ticks=0;
    char path[32];
    for(int j=0;j<maxpro;j++){
      sprintf(path,"/proc/%d/psinfo",j);
      if((f1=open(path,O_RDONLY))==-1){
        continue;
      }
      read(f1,buf,MAXLINE);
      parseline(buf,num_list);
      if(!strcmp(num_list[4],"R")){
        r_ticks+=atoi(num_list[7]);
      }
      a_ticks+=atoi(num_list[7]);
    }
    char c='%';
    printf("%d %d %d %lf%c",total_mem,free_mem,cached_mem,(double)(r_ticks)*100.0/(double)a_ticks,c);

    return;
  }

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

  if((dupset!=0&&count!=0)||(dupset!=0&&bg!=0)||(count!=0)&&bg!=0){
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

  // //执行管道指令
  // if (count!=0){
  //   int k=0;
  //   int fd[2];
  //   pipe(&fd[0]);

  //   if((Fork())==0){
  //       close(fd[0]);
  //       close(STD_OUTPUT);
  //       dup(fd[1]);
  //       close(fd[1]);
  //       Execve(argv[0],argv[0],NULL);
  //     }
  //     else{
  //       waitpid(-1,&status,0);
  //     }
  //     if((Fork())==0){
  //       close(fd[1]);
  //       close(STD_INPUT);
  //       dup(fd[0]);
  //       close(fd[0]);
  //       Execve(argv[i_list[k]],argv[i_list[k]],NULL);
  //     }
  //     else{
  //       waitpid(-1,&status,0);
  //     }

  //   while(i_list[k+1]!=-1){
  //     if((Fork())==0){
  //       close(fd[0]);
  //       close(STD_OUTPUT);
  //       dup(fd[1]);
  //       close(fd[1]);
  //       Execve(argv[i_list[k]],argv[i_list[k]],NULL);
  //     }
  //     else{
  //       waitpid(-1,&status,0);
  //     }
  //     if((Fork())==0){
  //       close(fd[1]);
  //       close(STD_INPUT);
  //       dup(fd[0]);
  //       close(fd[0]);
  //       Execve(argv[i_list[k+1]],argv[i_list[k+1]],NULL);
  //     }
  //     else{
  //       waitpid(-1,&status,0);
  //     }
  //     k+=1;
  //   }
  //   return;
  // }
  // //执行重定向到文件
  // if(dupset!=0){

  //   if(dupset==1){
  //     if((Fork())==0){
  //       close(STD_OUTPUT);
  //       int fd=open(filename,O_WRONLY);
  //       dup(fd);
  //       Execve(argv[0],argv,NULL);
  //     }
  //     else{
  //       waitpid(-1,&status,0);
  //       return;
  //     }
  //   }
  //   else if(dupset==2){
  //     if((Fork())==0){
  //       close(STD_OUTPUT);
  //       int fd=open(filename,O_APPEND);
  //       dup(fd);
  //       Execve(argv[0],argv,NULL);
  //     }
  //     else{
  //       waitpid(-1,&status,0);
  //       return;
  //     }
  //   }
  //   else{
  //     if((Fork())==0){
  //       close(STD_INPUT);
  //       int fd=open(filename,O_RDONLY);
  //       dup(fd);
  //       Execve(argv[0],argv,NULL);
  //     }
  //     else{
  //       waitpid(-1,&status,0);
  //       return;
  //     }
  //   }
  // }
  // // 执行后台运行
  // if(bg!=0){
  //   argv[argnum-1]="\0";
  //   if((Fork())==0){
  //     signal(SIGCHLD,SIG_IGN);
  //     int fd=open("/dev/null",O_RDONLY);

  //     dup2(fd,STDIN_FILENO);
  //     dup2(fd,STDOUT_FILENO);
  //     dup2(fd,STDERR_FILENO);

  //     Execve(argv[0],argv,NULL);
  //   }
  //   else{
  //     // waitpid(-1,&status,WNOHANG|WUNTRACED);
  //     return;
  //   }
  // }
  // // 执行前台运行

  // if((Fork())==0){
  //   Execve(argv[0],argv,NULL);
  // }
  // else{
  //   waitpid(-1,&status,0);
  //   return;
  // }
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
// pid_t Fork(void){
//   pid_t pid;

//   if ((pid = fork()) < 0)
//     unix_error("Fork error");
//   return pid;
// }

// void Execve(const char *filename, char *const argv[], char *const environ[]){
//   if (execve(filename, argv, environ) < 0) {
//     printf("%s: Command not found\n", argv[0]);
//     exit(0);
//   }
// }

// void unix_error(char *msg){
//     fprintf(stdout, "%s: %s\n", msg, strerror(errno));
//     exit(1);
// }