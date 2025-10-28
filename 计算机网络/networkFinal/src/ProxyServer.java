import java.io.*;
import java.net.*;
import java.util.Arrays;

public class ProxyServer {
    private static final int PORT = 8082;
    private ServerSocket serverSocket;

    public static void main(String[] args) {
        ProxyServer server = new ProxyServer();
        server.start();
    }

    public void start() {
        try {
            serverSocket = new ServerSocket(PORT);
            System.out.println("Proxy server started. Listening at port " + PORT);
            while (true) {
                Socket clientSocket = serverSocket.accept();
                new Thread(new ProxyClientHandler(clientSocket)).start();
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}

class ProxyClientHandler implements Runnable {
    private final Socket clientSocket;

    public ProxyClientHandler(Socket clientSocket) {
        this.clientSocket = clientSocket;
    }

    @Override
    public void run() {
        try {
            InputStream clientIn =clientSocket.getInputStream();
            BufferedReader in = new BufferedReader(new InputStreamReader(clientIn));
            OutputStream out = clientSocket.getOutputStream();
            StringBuilder head = new StringBuilder();
            String requestLine;
            String host = null;
            String method="";
            int len;
            int port = 80;
            while ((requestLine = in.readLine()) != null) {
                head.append(requestLine).append("\r\n");
                if (requestLine.length() == 0) {
                    break;
                }
                else {
                    String[] temp = requestLine.split(" ");
                    if (temp[0].contains("host") || temp[0].contains("Host")) {
                        host = temp[1];
                        if (host.contains(":")){
                            String[] tmp=host.split(":");
                            host=tmp[0];
                            port= Integer.parseInt(tmp[1]);
                        }
                    }
                }
                if (requestLine.trim().isEmpty()) {
                    break;
                }
            }
            head.append("\r\n");
//            System.out.println(head);
            try{
                method = head.substring(0, head.indexOf(" "));
            }
            catch(StringIndexOutOfBoundsException e){
                System.out.println("requestLine=NULL");
            }
//            System.out.println(method);
            Socket serverSocket = new Socket(host, port);
            System.out.println("连接到服务器：" + serverSocket.getInetAddress() + ":" + serverSocket.getPort());
            OutputStream serverOut = serverSocket.getOutputStream();
            InputStream serverIn = serverSocket.getInputStream();
            if (method.equals("CONNECT")) {
                out.write("HTTP/1.1 200 ConnectionEstablished\r\n\r\n".getBytes());
                out.flush();
                new request_handler(clientIn, serverOut).start();
            } else {
                serverOut.write(head.toString().getBytes());
            }
            byte[] returnBuffer = new byte[102400];
            while ((len = serverIn.read(returnBuffer)) != -1) {
                out.write(returnBuffer, 0, len);
            }
            in.close();
            out.flush();
            out.close();
            serverIn.close();
            serverOut.flush();
            serverOut.close();
            serverSocket.close();
            clientSocket.close();


        } catch (IOException e) {
            e.printStackTrace();
        }
    }
    private static class request_handler extends Thread
    {
        private final InputStream input;
        private final OutputStream output;
        public request_handler(InputStream input, OutputStream output)
        {
            this.input = input;
            this.output = output;
        }
        @Override
        public void run(){
            int len;
            byte[] buffer = new byte[204800];
            try {
                while ((len = input.read(buffer)) != -1) {
                    if (len > 0) {
                        output.write(buffer, 0, len);
                        output.flush();
                    }
                }
            }catch (IOException e){
                //System.out.println("转发/读取请求失败。");
            }
        }
    }
}