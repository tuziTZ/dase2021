package unit;
import consensus.MinerPeer;
import data.*;
import utils.SecurityUtil;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class UtxoTest {
    @org.junit.Test
    public void utxoTest(){
        BlockChain blockChain = new BlockChain();
        TransactionPool transactionPool = new TransactionPool(1);
        MinerPeer minerPeer = new MinerPeer(transactionPool, blockChain);
        Transaction transaction = getOneTransaction(blockChain);
        transactionPool.put(transaction);
        minerPeer.run();
    }
    Transaction getOneTransaction(BlockChain blockChain) {
//        生成一笔特殊交易：accounts[1]支付给accounts[2]1000元，accounts[1]使用自己的公钥对交易签名
//            可参考TransactionProducer中的getOneTransaction设计代码
    //TODO
        Random random = new Random(); // random.nextInt(bound) 在[0， bound) 中取
        Transaction transaction = null; // 返回的交易
        Account[] accounts = blockChain.getAccounts(); // 获取账户数组

        // 随机获取两个账户A和B
        Account aAccount = accounts[1];
        Account bAccount = accounts[2];
        // BTC不允许自己给自己转账

        // 获得钱包地址
        String aWalletAddress = aAccount.getWalletAddress();
        String bWalletAddress = bAccount.getWalletAddress();
        // 获取A可用的Utxo并计算余额
        UTXO[] aTrueUtxos = blockChain.getTrueUtxos(aWalletAddress);
        int aAmount = aAccount.getAmount(aTrueUtxos);
        //如果A账户的余额为0，则无法构建交易，重新随机生成

        // 随机生成交易数额 [1，aAmount] 之间
        int txAmount = 1000;
        // 构建InUtxo和0utUtxQ
        List<UTXO> inUtxoList = new ArrayList<>();
        List<UTXO> outUtxoList = new ArrayList<>();

        byte[] aUnlockSign = SecurityUtil.signature(aAccount.getPublicKey().getEncoded(), aAccount.getPrivateKey());
        // 选择输入总额>=交易数额的 utXQ
        int inAmount = 0;
        for (UTXO utxo : aTrueUtxos) {
            // 解锁成功才能使用该utxo
            if (utxo.unlockScript(aUnlockSign, aAccount.getPublicKey())) {
                inAmount += utxo.getAmount() ;
                inUtxoList.add(utxo);
                if (inAmount >= txAmount) {
                    break;
                }
            }
        }
        // 可解锁的utxo总额仍不足以支付交易数额，则重新随机

        // 构建输出0utUtxos，A账户向B账户支付txAmount，同时输入对方的公钥以供生成公钥哈希
        outUtxoList.add(new UTXO(bWalletAddress, txAmount, bAccount.getPublicKey()));
        //如果有余额，则“找零”，即给自己的utxo
        if (inAmount > txAmount) {
            outUtxoList.add(new UTXO(aWalletAddress,  inAmount - txAmount, aAccount.getPublicKey()));
        }
        //导出固定utxo数组
        UTXO[] inUtxos = inUtxoList.toArray(new UTXO[0]);
        UTXO[] outUtxos = outUtxoList.toArray(new UTXO[0]);
        //A账户需对整个交易进行私钥签名，确保交易不会被篡改，因为交易会传输到网络中，而上述步骤可在本地离线环境中构造
        // 获取要签名的数据，这个数据需要囊括交易信息
        byte[] data = SecurityUtil.utxos2Bytes(inUtxos, outUtxos);
        //A账户使用私钥签名
        byte[] sign = SecurityUtil.signature(data, aAccount.getPrivateKey());
        //交易时间戳
        long timestamp = System.currentTimeMillis();
        // 构造交易
        transaction = new Transaction(inUtxos, outUtxos, sign, aAccount.getPublicKey(), timestamp);

        return transaction;
    }
}
