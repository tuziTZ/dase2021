import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.net.InetSocketAddress;
import java.net.ServerSocket;
import java.net.Socket;
import java.util.Arrays;

public class RpcProvider {
    public static void main(String[] args) {
        Proxy2Impl proxy2Impl = new Proxy2Impl();
        try (ServerSocket serverSocket = new ServerSocket()) {
            serverSocket.bind(new InetSocketAddress(9091));
            try(Socket socket = serverSocket.accept()) {
// ObjectInputStream/ObjectOutStream 提供了将对象序列化和反序列化的功能
                ObjectInputStream is = new
                        ObjectInputStream(socket.getInputStream());
// rpc提供⽅和调⽤⽅之间协商的报⽂格式和序列化规则
                String methodName = is.readUTF();
                Class<?>[] parameterTypes = (Class<?>[]) is.readObject();
                int lengthArgs =  is.readInt();
                Object[] arguments = new Object[lengthArgs];
                for (int i = 0; i < lengthArgs; i++) {
                    arguments[i] = is.readObject();
                }
                System.out.println(arguments[0]);
                Object result;
                result = Proxy2Impl.class.getMethod(methodName,parameterTypes).invoke(proxy2Impl, arguments);
// rpc提供⽅调⽤本地的对象的⽅法

// 将结果序列化并返回
                new ObjectOutputStream(socket.getOutputStream()).writeObject(result);
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}