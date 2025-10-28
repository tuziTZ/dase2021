import java.io.*;
import java.net.ServerSocket;
import java.net.Socket;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;

//Task1
//Task2
public class TCPServer {
    private ServerSocket serverSocket;

    public void start(int port) throws IOException {
// 1. 创建⼀个服务器端Socket，即ServerSocket，监听指定端⼝
        serverSocket = new ServerSocket(port);
// 2. 调⽤accept()⽅法开始监听，阻塞等待客户端的连接
        for(;;){
            System.out.println("阻塞等待客户端连接中...");
            Socket clientSocket = serverSocket.accept();
            ClientHandler c= new ClientHandler(clientSocket);
            c.start();
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

class ClientHandler extends Thread {
    private Socket socket;

    ClientHandler(Socket socket) {
        this.socket = socket;
    }
    public char toUpperCase(char c1){
        int b = (int) c1 -32;
        return (char)b;
    }
    @Override
    public void run() {
        super.run();
        PrintWriter out = null;
        try {
            out = new PrintWriter(new OutputStreamWriter(this.socket.getOutputStream(),
                    StandardCharsets.UTF_8), true);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
        BufferedReader in = null;
        try {
            in = new BufferedReader(new InputStreamReader(this.socket.getInputStream(),
                    StandardCharsets.UTF_8));
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
        String str;

        while(true){
            try {
                if ((str = in.readLine()) == null) break;
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
            System.out.println("我是服务端，客户端说： " + str);
// 消息回写
            String newWord= "";
            char[] arrays = str.toCharArray();
            for (char c1 : arrays) {
                if (c1 >= 97 && c1 <= 122) {
                    char c2 = toUpperCase(c1);
                    newWord=String.join( "",newWord,Character.toString(c2));
                }
                else{
                    newWord=String.join( "",newWord,Character.toString(c1));
                }
            }
            out.println(newWord);
        }
    }
}





