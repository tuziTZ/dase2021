import java.io.*;
import java.net.Socket;
import java.nio.charset.StandardCharsets;
import java.util.Scanner;

public class TCPClient {
    private Socket clientSocket;
    private PrintWriter out;
    private BufferedReader in;
    public void startConnection(String ip, int port) throws IOException {
// 1. 创建客户端Socket，指定服务器地址，端⼝
        clientSocket = new Socket(ip, port);
// 2. 获取输⼊输出流
        out = new PrintWriter(new OutputStreamWriter(clientSocket.getOutputStream(),
                StandardCharsets.UTF_8), true);
        in = new BufferedReader(new InputStreamReader(clientSocket.getInputStream(),
                StandardCharsets.UTF_8));
    }
    public void sendMessage(String msg) throws IOException {
// 3. 向服务端发送消息
        out.println(msg);
// 4. 接收服务端回写信息
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
        try {
            client.startConnection("127.0.0.1", port);
            Scanner sc=new Scanner(System.in);
            while(true){
                String myWord=sc.next();
                client.sendMessage(myWord);
            }
        }catch (IOException e){
            e.printStackTrace();
        } finally {
            client.stopConnection();
        }
    }
}

