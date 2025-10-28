package data;

import utils.SecurityUtil;

import java.security.PublicKey;
import java.util.Arrays;

/**
 * 对交易的抽象
 */
public class Transaction {

    private final UTXO[] inUtxos;
    private final UTXO[] outUtxos;

    private final byte[] sendSign;
    private final PublicKey sendPublicKey;
    private final long timestamp;

//    private final String data;
//    private final long timestamp;

    public Transaction(UTXO[] inUtxos,UTXO[] outUtxos, byte[] sendSign,PublicKey sendpublicKey, long timestamp) {
        this.inUtxos = inUtxos;
        this.outUtxos = outUtxos;
        this.sendSign=sendSign;
        this.sendPublicKey=sendpublicKey;
        this.timestamp = timestamp;
    }

//    public String getData() {
//        return data;
//    }

    public long getTimestamp() {
        return timestamp;
    }

    public UTXO[] getInUtxos() {
        return inUtxos;
    }

    public UTXO[] getOutUtxos() {
        return outUtxos;
    }

    public byte[] getSendSign() {
        return sendSign;
    }

    public PublicKey getSendPublicKey() {
        return sendPublicKey;
    }

    @Override
    public String toString() {
        return "Transaction{" +
                "inUtxos=" + Arrays.toString(inUtxos) +
                ", outUtxos=" + Arrays.toString(outUtxos) +
                ", sendSign=" + SecurityUtil.bytes2HexString(sendSign) +
                ", sendPublicKey=" + SecurityUtil.bytes2HexString(sendPublicKey.getEncoded()) +
                ", timestamp=" + timestamp +
                '}';
    }
}
