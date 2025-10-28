#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <unistd.h>
#include <sys/time.h>
#include<sys/wait.h>
#include <sys/types.h>    
#include <sys/stat.h>    
#include <fcntl.h>
#include <string.h>

#define times 500//随机or顺序读写的次数
#define filesize (300*1024*1024)//文件大小300MB
#define readsize (1024*1024*1024)

char *filePathDisk[17]={"/usr/file1.txt","/usr/file2.txt","/usr/file3.txt","/usr/file4.txt","/usr/file5.txt","/usr/file6.txt","/usr/file7.txt","/usr/file8.txt","/usr/file9.txt","/usr/file10.txt","/usr/file11.txt","/usr/file12.txt","/usr/file13.txt","/usr/file14.txt","/usr/file15.txt","/usr/file16.txt","/usr/file17.txt"};
char *filePathRam[17]={"/root/myram/file1.txt","/root/myram/file2.txt","/root/myram/file3.txt","/root/myram/file4.txt","/root/myram/file5.txt","/root/myram/file6.txt","/root/myram/file7.txt","/root/myram/file8.txt","/root/myram/file9.txt","/root/myram/file10.txt","/root/myram/file11.txt","/root/myram/file12.txt","/root/myram/file13.txt","/root/myram/file14.txt","/root/myram/file15.txt","/root/myram/file16.txt","/root/myram/file17.txt"};
int bs[6]={64,256,1024,4096,16384,65536};
char buf[65536];
char readbuf[readsize];

void write_file(int blocksize, bool isrand, char *filepath, int fs){
  int fp=open(filepath, O_WRONLY|O_SYNC|O_CREAT);
  int res;
  if(fp == -1){
    printf("open file error\n");
    return;
  }
  for(int i=0;i<times;i++){
    if((res=write(fp, buf, blocksize))!=blocksize){
      printf("write file error num %d, finished %d times\n",res,i+1);
      // perror("error:");
      return;
    }
    if(isrand){
      // printf("fs=%d,blocksize=%d\n",fs,blocksize);
      lseek(fp, rand()%(fs-blocksize), SEEK_SET);
    }
  }
  lseek(fp, 0, SEEK_SET);
  close(fp);
}
void read_file(int blocksize, bool isrand, char *filepath, int fs){
  int fp=open(filepath, O_RDONLY|O_SYNC|O_CREAT);
  int res;
  if(fp == -1){
    printf("open file error\n");
    return;
  }
  for(int i=0;i<times;i++){
    if((res=read(fp, readbuf, blocksize))!=blocksize){
      printf("read file error num %d\n",res);
      return;
    }
    if(isrand){
      // printf("fs=%d,blocksize=%d\n",fs,blocksize);
      lseek(fp, rand()%(fs-blocksize), SEEK_SET);
    }
  }
  lseek(fp, 0, SEEK_SET);
  close(fp);

}
long get_time_left(long starttime, long endtime){
  //返回微秒
  long spendtime;
  spendtime=endtime-starttime;
  return spendtime;
}
int main(){
  // printf("ram 顺序 写\n");
  // printf("ram 顺序 读\n");
  // printf("ram 随机 写\n");
  // printf("ram 随机 读\n");
  // printf("disk 顺序 写\n");
  printf("disk 顺序 读\n");
  // printf("disk 随机 写\n");
  // printf("disk 随机 读\n");
  int concurrency=7;
  memset(buf,0,sizeof(buf));
  struct timeval st,et;
  //测试concurrency对读写速度的影响（使用ram,顺序写,块大小固定为1024）
  for(int j=0;j<6;j++){
    // int j=3;
    int blocksize=bs[j];
    printf("blocksize=%d\n",blocksize);
    // for(concurrency=1;concurrency<=16;concurrency++){
      gettimeofday(&st,NULL);
      for(int i=0;i<concurrency;i++){
        if(fork()==0){
          // write_file(blocksize,false,filePathRam[i],filesize/concurrency);
          // read_file(blocksize,false,filePathRam[i],filesize/concurrency);
          // write_file(blocksize,true,filePathRam[i],filesize/concurrency);
          // read_file(blocksize,true,filePathRam[i],filesize/concurrency);
          // write_file(blocksize,false,filePathDisk[i],filesize/concurrency);
          read_file(blocksize,false,filePathDisk[i],filesize/concurrency);
          // write_file(blocksize,true,filePathDisk[i],filesize/concurrency);
          // read_file(blocksize,true,filePathDisk[i],filesize/concurrency);
          exit(0);
        }
      }
      while(wait(NULL)!=-1){

      }
      gettimeofday(&et,NULL);
      // long spendtime=get_time_left(st.tv_usec,et.tv_usec);//花费的时间
      long spendtime=(et.tv_sec-st.tv_sec)*1000+(et.tv_usec-st.tv_usec)/1000;//ms
      // printf("%d %d\n",st.tv_usec,et.tv_sec);
      double speed=blocksize*concurrency*times/(spendtime*1000.0);//吞吐量
      printf("speed=%.2fMB/s,spendtime=%ldms,concurrency=%d\n",speed,spendtime,concurrency);
    // }

  }
  
  return 0;
}