public class Test {
    public static void main(String[] args) {
        ThreadTest thread1 = new ThreadTest();
        ThreadTest thread2 = new ThreadTest();
        thread1.start();
        thread2.start();
    }
}
