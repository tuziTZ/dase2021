public class ThreadTest04 implements Runnable{
    @Override
    public void run(){
        int worktime = 0;
        while(true){
            System.out.println("助教在教室的第"+ worktime +"秒");
            try{
                Thread.currentThread().sleep(1000);
            }catch (InterruptedException e){
                e.printStackTrace();
            }
            worktime ++;
        }
    }
    public static void main(String[] args) throws InterruptedException{
// TODO
        ThreadTest04 assist = new ThreadTest04();
        Thread thread4=new Thread(assist,"助教线程");
        thread4.setDaemon(true);
        thread4.start();
        for(int i = 0; i < 10; i++){
            Thread.sleep(1000);
            System.out.println("同学们正在上课");
            if(i == 9){
                System.out.println("同学们下课了");
            }
        }}
}