import java.io.*;
import java.net.ServerSocket;
import java.net.Socket;
import java.nio.charset.StandardCharsets;
public class TCPServer {
    private ServerSocket serverSocket;
    private Socket clientSocket;
    public void start(int port) throws IOException {
        serverSocket = new ServerSocket(port);
        System.out.println("阻塞等待客户端连接中...");
        clientSocket = serverSocket.accept();
        InputStream is = clientSocket.getInputStream();
        File file = new File("checksum_recv.txt");
        FileOutputStream output = new FileOutputStream(file);
        while(true){
            int data;
            data = is.read();
            if(data==-1){
                break;
            }
//            System.out.println("发送数据报为："+data);
            output.write(data);
        }
    }
    public void stop(){
        try {
            if(clientSocket!=null) clientSocket.close();
            if(serverSocket!=null) serverSocket.close();
        }catch (IOException e){
            e.printStackTrace();
        }
    }
    public static void main(String[] args) throws IOException {
        int port = 9091;
        TCPServer server=new TCPServer();
        server.start(port);
        server.stop();
        System.out.println("接收⽂件已完成");
        File file = new File("checksum_recv.txt");
        System.out.println("接收⽂件的md5为: " + MD5Util.getMD5(file));

    }
}