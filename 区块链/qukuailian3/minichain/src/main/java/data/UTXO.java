package data;

import utils.SecurityUtil;

import java.security.PublicKey;
import java.util.Arrays;
import java.util.Stack;

public class UTXO {
    private final  String walletAddress;//交易方获得的钱包地址
    private final int amount;//比特币数量
    private final byte[] publicKeyHash;//交易获得方的公钥
    public UTXO(String walletAddress, int amount, PublicKey publicKey){
        this.walletAddress=walletAddress;
        this.amount=amount;
        //对公钥进行哈希摘要，作为解锁脚本数据
        publicKeyHash= SecurityUtil.ripemd160Digest(SecurityUtil.sha256Digest(publicKey.getEncoded()));
    }
    public boolean unlockScript(byte[] sign, PublicKey publicKey){
        Stack<byte[]> stack = new Stack<>();
        //<sig>签名入栈
        stack.push(sign);
        //<PubK>公钥入栈
        stack.push(publicKey.getEncoded());
        //复制一份栈顶数据peek
        stack.push(stack.peek());
        //对公钥进行哈希摘要
        byte[] data=stack.pop();
        stack.push(SecurityUtil.ripemd160Digest(SecurityUtil.sha256Digest(data)));
        stack.push(publicKeyHash);
        byte[] publicKeyHash1 = stack.pop();
        byte[] publicKeyHash2 = stack.pop();
        if(!Arrays.equals(publicKeyHash1,publicKeyHash2)){
            return false;
        }
        byte[] publicKeyEncoded=stack.pop();
        byte[] sign1=stack.pop();
        return SecurityUtil.verify(publicKey.getEncoded(),sign1,publicKey);

    }

    public String getWalletAddress() {
        return walletAddress;
    }

    public int getAmount() {
        return amount;
    }

    public byte[] getPublicKeyHash() {
        return publicKeyHash;
    }

    @Override
    public String toString() {
        return "\n\tUTXO{" +
                "walletAddress='" + walletAddress + '\'' +
                ", amount=" + amount +
                ", publicKeyHash=" + SecurityUtil.bytes2HexString(publicKeyHash) +
                '}';
    }

}
