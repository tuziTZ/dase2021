public class TestInteract {
    public static void main(String[] args) {
        PlusMinusOne pmo = new PlusMinusOne();
        pmo.num = 50;
        Thread t1 = new Thread() {
            public void run() {
                while (true) {
                    synchronized (pmo){

                        while (pmo.num == 1) {
                            continue;
                        }
                        pmo.minusOne();
                    }
                    try {
                        Thread.sleep(10);
                    } catch (InterruptedException e) {
                        e.printStackTrace();
                    }
                }
            }
        };
        t1.start();
        Thread t3 = new Thread() {
            public void run() {
                while (true) {
                    synchronized (pmo){

                        while (pmo.num == 1) {
                            continue;
                        }
                        pmo.minusOne();
                    }
                    try {
                        Thread.sleep(10);
                    } catch (InterruptedException e) {
                        e.printStackTrace();
                    }
                }
            }
        };
        t3.start();
        Thread t2 = new Thread() {
            public void run() {
                while (true) {
                    pmo.plusOne();
                    try {
                        Thread.sleep(100);
                    } catch (InterruptedException e) {
                        e.printStackTrace();
                    }
                }
            }
        };
        t2.start();
    }
}