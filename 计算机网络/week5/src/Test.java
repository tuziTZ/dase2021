import java.util.LinkedList;
import java.util.Queue;
public class Test {
    public static void main(String[] args) {
        ProductFactory pf = new ProductFactory();
        Thread t1 = new Thread() {
            public void run() {
                while (true) {
                    try {
                        String s = pf.getProduct();
                        System.out.println("t1 get product: " + s);
                        Thread.sleep(100);
                    } catch (InterruptedException e) {
                        e.printStackTrace();
                    }
                }
            }
        };
        Thread[] ta = new Thread[10];
        for(int i=0;i<10;i++){
            int finalI = i;
            ta[i]=new Thread(){
                public void run() {
                    while (true) {
                        try {
                            String s = pf.getProduct();
                            System.out.println("t1-"+ finalI +" get product: " + s);
                            Thread.sleep(100);
                        } catch (InterruptedException e) {
                            e.printStackTrace();
                        }
                    }
                }
            };
        }
        for(int i=0;i<10;i++) {
            ta[i].start();
        }
        Thread t2 = new Thread() {
            public void run() {
                while (true) {
                    try {
                        String s = "product";
                        pf.addProduct(s);
                        System.out.println("t2 add product: " + s);
                        Thread.sleep(1000);
                    } catch (InterruptedException e) {
                        e.printStackTrace();
                    }
                }
            }
        };
        t1.start();
        t2.start();
    }
}
class ProductFactory {

    Queue<String> productQueue = new LinkedList<>();
    public synchronized void addProduct(String s) {
        productQueue.add(s);
        this.notify(); // 唤醒所有在this锁等待的线程
    }
    public synchronized String getProduct() throws InterruptedException {
        if (productQueue.isEmpty()) {
            Thread t = Thread.currentThread();
            long id = t.threadId();
            System.out.println("线程"+id+"进入等待");
// 释放this锁
            this.wait();
// 重新获取this锁
        }
        return productQueue.remove();
    }
}