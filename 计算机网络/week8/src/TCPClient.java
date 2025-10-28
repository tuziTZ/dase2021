import java.io.*;
import java.net.Socket;

import java.nio.charset.StandardCharsets;
import java.util.Scanner;

public class TCPClient {
    private Socket clientSocket;
    private PrintWriter out;
    private BufferedReader in;
    private Scanner sc;
    private ServerReadHandler serverReadHandler;
    private ServerWriteHandler serverWriteHandler;
    private ServerHandler serverHandler;
    public void startConnection(String ip, int port) throws IOException {
// 1. 创建客户端Socket，指定服务器地址，端⼝
        clientSocket = new Socket(ip, port);
// 2. 获取输⼊输出流
        out = new PrintWriter(new OutputStreamWriter(clientSocket.getOutputStream(),
                StandardCharsets.UTF_8), true);
        in = new BufferedReader(new InputStreamReader(clientSocket.getInputStream(),
                StandardCharsets.UTF_8));

        sc = new Scanner(System.in);
        serverReadHandler = new ServerReadHandler();
        serverWriteHandler = new ServerWriteHandler();
        serverHandler=new ServerHandler();
    }

    class ServerReadHandler extends Thread {
        @Override
        public void run() {
            try {
                while (true) {
// 拿到服务器⼀条数据，显示在控制台中
                    String resp = in.readLine();
                    if (resp == null) {
                        System.out.println("从服务器读到的数据为空");
                        break;
                    } else {
                        System.out.println("从服务器读到的数据为：" + resp);
                    }
                }
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }
    class ServerWriteHandler extends Thread {
        //从控制台拿到数据，发送给服务器
        void send(String str){
            out.println(str);
        }
        @Override
        public void run() {
            while (sc.hasNext()) {
                String str = sc.next();
                send(str);
            }
        }
    }
    class ServerHandler extends Thread {
        @Override
        public void run() {
            super.run();
            serverReadHandler.start();
            serverWriteHandler.start();
            try {
                serverReadHandler.join();
                serverWriteHandler.join();
            } catch (InterruptedException e) {
                throw new RuntimeException(e);
            }

        }
    }


    public void stopConnection() {
// 关闭相关资源
        try {
            if(in!=null) in.close();
            if(out!=null) out.close();
            if(clientSocket!=null) clientSocket.close();
        } catch (IOException e){
            e.printStackTrace();
        }
    }
    public static void main(String[] args) {
        int port = 9091;
        TCPClient client = new TCPClient();
//        try {
//            client.startConnection("127.0.0.1", port);
//            client.serverHandler.start();
//            client.serverHandler.join();
//        }catch (IOException e){
//            e.printStackTrace();
//        } catch (InterruptedException e) {
//            throw new RuntimeException(e);
//        } finally {
//            client.stopConnection();
//        }
        try{
            client.startConnection("127.0.0.1", port);
            client.serverHandler.start();
//            client.serverHandler.join();
            Thread.sleep(100000);
        } catch (IOException | InterruptedException e) {
            throw new RuntimeException(e);
        }finally {
            client.stopConnection();
        }
    }
}



