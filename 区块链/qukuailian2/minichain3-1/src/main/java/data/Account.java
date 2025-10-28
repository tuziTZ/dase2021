package data;

import utils.Base58Util;
import utils.SecurityUtil;
import java.security.KeyPair;
import java.security.PrivateKey;
import java.security.PublicKey;

public class Account {
    private final PublicKey publicKey;
    private final PrivateKey privateKey;
    public Account(){
        KeyPair keyPair = SecurityUtil.secp256k1Generate();
        this.privateKey=keyPair.getPrivate();
        this.publicKey=keyPair.getPublic();

    }
    // 根据账户公钥计算钱包地址
    public String getWalletAddress(){
        //公钥哈希：RIPEMD160(SHA256(PubK)
        byte[] publicKeyHash=SecurityUtil.ripemd160Digest(SecurityUtil.sha256Digest(publicKey.getEncoded()));
        // 0x00+公钥哈希
        byte[] data = new byte[1+publicKeyHash.length];
        data[0]=(byte)0;
        for(int i=0;i<publicKeyHash.length;++i){
            data[i+1]= publicKeyHash[i];

        }
        //两次sha256哈希摘要
        byte[] doubleHash=SecurityUtil.sha256Digest(SecurityUtil.sha256Digest(data));
        //0x00+公钥哈希+校验（两次哈希后前4字节）
        byte[] walletEncoded=new byte[1+publicKeyHash.length+4];
        walletEncoded[0] =(byte)0;
        for(int i=0;i<publicKeyHash.length;++i){
            walletEncoded[1+i]=publicKeyHash[i];
        }
        for (int i=0;i<4;++i){
            walletEncoded[1+publicKeyHash.length+i]=doubleHash[i];
        }
        String walletAddress= Base58Util.encode(walletEncoded);
        return walletAddress;
    }
    //根据与该账户关联的未使用的utxo计算账户余额
    public int getAmount(UTXO[] trueUtxos){
        int amount = 0;
        for(int i=0;i<trueUtxos.length;++i){
            amount+=trueUtxos[i].getAmount();
        }
        return amount;
    }

    public PublicKey getPublicKey() {
        return publicKey;
    }

    public PrivateKey getPrivateKey() {
        return privateKey;
    }

    @Override
    public String toString() {
        return "Account{" +
                "publicKey=" + SecurityUtil.bytes2HexString(publicKey.getEncoded()) +
                ", privateKey=" + SecurityUtil.bytes2HexString(privateKey.getEncoded()) +
                '}';
    }
}
