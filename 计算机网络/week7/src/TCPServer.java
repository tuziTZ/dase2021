import java.io.*;
import java.net.ServerSocket;
import java.net.Socket;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

//Task1
//public class TCPServer {
//    private ServerSocket serverSocket;
//    private Socket clientSocket;
//    private PrintWriter out;
//    private BufferedReader in;
//    public void start(int port) throws IOException {
//// 1. 创建⼀个服务器端Socket，即ServerSocket，监听指定端⼝
//        serverSocket = new ServerSocket(port);
//// 2. 调⽤accept()⽅法开始监听，阻塞等待客户端的连接
//        System.out.println("阻塞等待客户端连接中...");
//        clientSocket = serverSocket.accept();
//// 3. 获取Socket的字节输出流
//        out = new PrintWriter(new OutputStreamWriter(clientSocket.getOutputStream(),
//                StandardCharsets.UTF_8), true);
//// 4. 获取Socket的字节输⼊流，并准备读取客户端发送的信息
//        in = new BufferedReader(new InputStreamReader(clientSocket.getInputStream(),
//                StandardCharsets.UTF_8));
//// 5. 阻塞读取客户端发送的信息
//        String str;
//
//        while((str = in.readLine())!= null){
//            System.out.println("我是服务端，客户端说： " + str);
//// 消息回写
//            String newWord= "";
//            char[] arrays = str.toCharArray();
//            for (char c1 : arrays) {
//                if (c1 >= 97 && c1 <= 122) {
//                    char c2 = toUpperCase(c1);
//                    newWord=String.join( "",newWord,Character.toString(c2));
//                }
//                else{
//                    newWord=String.join( "",newWord,Character.toString(c1));
//                }
//            }
//            out.println(newWord);
//        }
//    }
//    public char toUpperCase(char c1){
//        int b = (int) c1 -32;
//        return (char)b;
//    }
//    public void stop(){
//// 关闭相关资源
//        try {
//            if(in!=null) in.close();
//            if(out!=null) out.close();
//            if(clientSocket!=null) clientSocket.close();
//            if(serverSocket!=null) serverSocket.close();
//        }catch (IOException e){
//            e.printStackTrace();
//        }
//    }
//    public static void main(String[] args) {
//        int port = 9091;
//        TCPServer server=new TCPServer();
//        try {
//            server.start(port);
//        }catch (IOException e){
//            e.printStackTrace();
//        }finally {
//            server.stop();
//        }
//    }
//}






//Task2
//public class TCPServer {
//    private ServerSocket serverSocket;
//
//    public void start(int port) throws IOException {
//// 1. 创建⼀个服务器端Socket，即ServerSocket，监听指定端⼝
//        serverSocket = new ServerSocket(port);
//// 2. 调⽤accept()⽅法开始监听，阻塞等待客户端的连接
//        for(;;){
//            System.out.println("阻塞等待客户端连接中...");
//            Socket clientSocket = serverSocket.accept();
//            ClientHandler c= new ClientHandler(clientSocket);
//            c.start();
//        }
//    }
//    public void stop(){
//// 关闭相关资源
//        try {
//            if(serverSocket!=null) serverSocket.close();
//        } catch (IOException e){
//            e.printStackTrace();
//        }
//    }
//    public static void main(String[] args) {
//        int port = 9091;
//        TCPServer server=new TCPServer();
//        try {
//            server.start(port);
//        }catch (IOException e){
//            e.printStackTrace();
//        }finally {
//            server.stop();
//        }
//    }
//}
//
//class ClientHandler extends Thread {
//    private Socket socket;
//
//    ClientHandler(Socket socket) {
//        this.socket = socket;
//    }
//    public char toUpperCase(char c1){
//        int b = (int) c1 -32;
//        return (char)b;
//    }
//    @Override
//    public void run() {
//        super.run();
//        PrintWriter out = null;
//        try {
//            out = new PrintWriter(new OutputStreamWriter(this.socket.getOutputStream(),
//                    StandardCharsets.UTF_8), true);
//        } catch (IOException e) {
//            throw new RuntimeException(e);
//        }
//        BufferedReader in = null;
//        try {
//            in = new BufferedReader(new InputStreamReader(this.socket.getInputStream(),
//                    StandardCharsets.UTF_8));
//        } catch (IOException e) {
//            throw new RuntimeException(e);
//        }
//        String str;
//
//        while(true){
//            try {
//                if ((str = in.readLine()) == null) break;
//            } catch (IOException e) {
//                throw new RuntimeException(e);
//            }
//            System.out.println("我是服务端，客户端说： " + str);
//// 消息回写
//            String newWord= "";
//            char[] arrays = str.toCharArray();
//            for (char c1 : arrays) {
//                if (c1 >= 97 && c1 <= 122) {
//                    char c2 = toUpperCase(c1);
//                    newWord=String.join( "",newWord,Character.toString(c2));
//                }
//                else{
//                    newWord=String.join( "",newWord,Character.toString(c1));
//                }
//            }
//            out.println(newWord);
//        }
//    }
//}


//Task3
public class TCPServer {
    private ServerSocket serverSocket;
    private Socket clientSocket;
    private static int BYTE_LENGTH = 64;
    public void start(int port) throws IOException {
        serverSocket = new ServerSocket(port);
        System.out.println("阻塞等待客户端连接中...");
        clientSocket = serverSocket.accept();
        InputStream is = clientSocket.getInputStream();
        byte[] readBuffer = new byte[1024];
        int bufferCount=0;
        for(;;) {
            byte[] tmp=new byte[BYTE_LENGTH];
            int cnt = is.read(tmp, 0, BYTE_LENGTH);
            if (cnt>0)
                System.arraycopy(tmp,0,readBuffer,bufferCount,cnt);
            bufferCount+=cnt;
            if (bufferCount<=1){
                continue;
            }
            int length=readBuffer[0];
            if(bufferCount<1+length){
                continue;
            }
            byte[] content=Arrays.copyOfRange(readBuffer, 1, length + 1);
            System.out.println("服务端已收到消息: " + new String(content).trim());
            bufferCount-=(1+length);
            System.arraycopy(readBuffer,length+1,readBuffer,0,bufferCount);

        }
    }
    public void stop(){
// 关闭相关资源
        try {
            if(clientSocket!=null) clientSocket.close();
            if(serverSocket!=null) serverSocket.close();
        }catch (IOException e){
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
