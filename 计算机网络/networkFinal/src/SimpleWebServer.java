import java.io.*;
import java.net.*;

public class SimpleWebServer {
    private static final int PORT = 8081;
    private ServerSocket serverSocket;

    public static void main(String[] args) {
        SimpleWebServer server = new SimpleWebServer();
        server.start();
    }

    public void start() {
        try {
            serverSocket = new ServerSocket(PORT);
            System.out.println("Server started. Listening at port " + PORT);
            while (true) {
                Socket clientSocket = serverSocket.accept();
                new Thread(new ClientHandler(clientSocket)).start();
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}

class ClientHandler implements Runnable {
    private Socket clientSocket;

    public ClientHandler(Socket clientSocket) {
        this.clientSocket = clientSocket;
    }

    @Override
    public void run() {
        try {
            BufferedReader in = new BufferedReader(new InputStreamReader(clientSocket.getInputStream()));
            OutputStream out = clientSocket.getOutputStream();

            String requestLine = in.readLine();
            System.out.println(requestLine);

            String[] tokens = requestLine.split(" ");
            String method = tokens[0].toUpperCase();
            String resource = tokens[1];

            if (method.equals("GET")||method.equals("POST")) {
                if (resource.contains("/index.html")) {
                    File file = new File("D:/Javaprojects/networkFinal/src/index.html");
                    if (file.exists()) {
                        out.write("HTTP/1.1 200 OK\r\n".getBytes());
                        out.write("Content-Type: text/html; charset=UTF-8\r\n".getBytes());
                        out.write("\r\n".getBytes());

                        FileInputStream fileInputStream = new FileInputStream(file);
                        byte[] buffer = new byte[1024];
                        int len;
                        while ((len = fileInputStream.read(buffer)) != -1) {
                            out.write(buffer, 0, len);
                        }
                        fileInputStream.close();
                    } else {
                        out.write("HTTP/1.1 404 Not Found\r\n".getBytes());
                        out.write("Content-Type: text/plain; charset=UTF-8\r\n".getBytes());
                        out.write("\r\n".getBytes());
                        out.write("File not found.".getBytes());
                    }
                } else if (resource.equals("/shutdown")) {
                    out.write("HTTP/1.1 200 OK\r\n".getBytes());
                    out.write("Content-Type: text/plain; charset=UTF-8\r\n".getBytes());
                    out.write("\r\n".getBytes());
                    out.write("Server is shutting down...".getBytes());
                    clientSocket.close();
                    System.exit(0);
                } else {
                    out.write("HTTP/1.1 404 Not Found\r\n".getBytes());
                    out.write("Content-Type: text/plain; charset=UTF-8\r\n".getBytes());
                    out.write("\r\n".getBytes());
                    out.write("404 not found.".getBytes());
                }
            }
            else {
                out.write("HTTP/1.1 405 Method Not Allowed\r\n".getBytes());
                out.write("Content-Type: text/plain; charset=UTF-8\r\n".getBytes());
                out.write("\r\n".getBytes());
                out.write("Method not allowed.".getBytes());
            }

            in.close();
            out.flush();
            out.close();
            clientSocket.close();

        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}


