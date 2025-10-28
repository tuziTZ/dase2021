import java.io.*;
import java.net.ServerSocket;
import java.net.Socket;
import java.nio.charset.StandardCharsets;

import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;


//Task2
public class TCPServer {
    private static final List<Socket> list=new ArrayList<>();
    private ServerSocket serverSocket;

    public void start(int port) throws IOException {
// 1. 创建⼀个服务器端Socket，即ServerSocket，监听指定端⼝
        serverSocket = new ServerSocket(port);
// 2. 调⽤accept()⽅法开始监听，阻塞等待客户端的连接
        for (;;) {
            Socket socket = serverSocket.accept();
            SendToClient stc = new SendToClient(socket);
            stc.start();
            list.add(socket);

        }
    }

    static class SendToClient extends Thread {
        private final PrintWriter printWriter;
        private final Socket socket;
        private int len;
        SendToClient(Socket socket) throws IOException{
            this.socket=socket;
            this.printWriter = new PrintWriter(new OutputStreamWriter(socket.getOutputStream(),
                    StandardCharsets.UTF_8), true);
            this.len=0;
        }
        void send(String str){
            this.printWriter.println(str);
        }
        @Override
        public void run() {
            try {
                CheckClient cc= new CheckClient(socket);
                cc.start();
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
            while(true){
                synchronized (list){
                    if (list.size()!=len){
                        for (Socket value : list) {
                            send(value.getRemoteSocketAddress().toString());
                        }
                        len=list.size();
                    }
                }
            }
        }
    }
    static class CheckClient extends Thread {
        private final BufferedReader bufferedReader;
        private final Socket socket;
        CheckClient(Socket socket) throws IOException {
            InputStream inputStream = socket.getInputStream();
            this.bufferedReader = new BufferedReader(new InputStreamReader(inputStream,
                    StandardCharsets.UTF_8));
            this.socket=socket;
        }
        @Override
        public void run() {
            try {
                while (true) {
                    bufferedReader.readLine();
                }
            } catch (IOException e) {
                synchronized (list){
                    list.remove(socket);
                }
            }
        }
    }

    public void stop(){
// 关闭相关资源
        try {
            if(serverSocket!=null) serverSocket.close();
        } catch (IOException e){
            e.printStackTrace();
        }
    }
    public static void main(String[] args) {
        int port = 9091;
        TCPServer server=new TCPServer();
        try {
            server.start(port);
        }catch (IOException e){
            e.printStackTrace();
        }finally {
            server.stop();
        }
    }
}




// 处理从客户端读数据的线程
class ClientReadHandler extends Thread {
    private final BufferedReader bufferedReader;
    ClientReadHandler(InputStream inputStream) {
        this.bufferedReader = new BufferedReader(new InputStreamReader(inputStream,
                StandardCharsets.UTF_8));
    }
    @Override
    public void run() {
        try {
            while (true) {
// 拿到客户端⼀条数据
                String str = bufferedReader.readLine();
                if (str == null) {
                    System.out.println("从客户端读到的数据为空");
                    break;
                } else {
                    System.out.println("从客户端读到的数据为：" + str);
                }
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
// 处理向客户端写数据的线程
class ClientWriteHandler extends Thread {
    private final PrintWriter printWriter;
    private final Scanner sc;
    ClientWriteHandler(OutputStream outputStream) {
        this.printWriter = new PrintWriter(new OutputStreamWriter(outputStream,
                StandardCharsets.UTF_8), true);
        this.sc = new Scanner(System.in);
    }
    void send(String str){
        this.printWriter.println(str);
    }
    @Override
    public void run() {
        while (sc.hasNext()) {
// 拿到控制台数据
            String str = sc.next();
            send(str);
        }
    }
}

class ClientHandler extends Thread {
    private final ClientReadHandler clientReadHandler;
    private final ClientWriteHandler clientWriteHandler;
    ClientHandler(Socket socket) throws IOException{
        this.clientReadHandler = new ClientReadHandler(socket.getInputStream());
        this.clientWriteHandler = new ClientWriteHandler(socket.getOutputStream());
    }
    @Override
    public void run() {
        super.run();
        clientReadHandler.start();
        clientWriteHandler.start();
    }
}


