package unit;

import consensus.MinerNode;
import data.*;
import utils.SecurityUtil;

import java.security.PublicKey;
import java.util.ArrayList;

/**
 * @ClassName: UtxoTest
 * @Description: TODO
 * @Author xtm
 * @Date: 2022/3/4 15:23
 * @Version 1.0
 */
public class UtxoTest {

    @org.junit.Test
    public void utxoTest() {
        // 相关类初始化
        BlockChain blockChain = new BlockChain();   // 初始一个链，内部会创建accounts
        TransactionPool transactionPool = new TransactionPool(1);   // 交易池有一个交易就会被矿工打包
        MinerNode minerNode = new MinerNode(transactionPool, blockChain);

        // 生成一笔特殊交易
        Transaction transaction = getOneTransaction(blockChain);
        // 将该交易放入交易池中
        transactionPool.put(transaction);
        // 矿工工作
        minerNode.run();
    }

    /**
     * 生成一笔特殊交易：accounts[1] 支付给 accounts[2] 1000元, accounts[1]使用自己的公钥对交易签名
     * @param blockChain
     * @return
     */
    Transaction getOneTransaction(BlockChain blockChain) {
        Transaction transaction = null;
        Account[] accounts = blockChain.getAccounts();
        Account accountA = accounts[1];
        Account accountB = accounts[2];
        int txAmount = 1000;

        byte[] sign = SecurityUtil.signature(accountA.getPublicKey().getEncoded(), accountA.getPrivateKey());
        UTXO[] trueUtxos = blockChain.getTrueUtxos(accountA.getWalletAddress());

        ArrayList<UTXO> inUtxoList = new ArrayList<>();
        ArrayList<UTXO> outUtxoList = new ArrayList<>();

        int inAmount = 0;
        for (UTXO utxo: trueUtxos) {
            if (utxo.unlockScript(sign, accountA.getPublicKey())) {
                inAmount += utxo.getAmount();
                inUtxoList.add(utxo);
                if (inAmount >= txAmount) {
                    break;
                }
            }
        }

        outUtxoList.add(new UTXO(accountB.getWalletAddress(), txAmount, accountB.getPublicKey()));
        if (inAmount > txAmount) {
            outUtxoList.add(new UTXO(accountA.getWalletAddress(), inAmount - txAmount, accountA.getPublicKey()));
        }

        UTXO[] inUtxos = inUtxoList.toArray(new UTXO[0]);
        UTXO[] outUtxos = outUtxoList.toArray(new UTXO[0]);
        byte[] data = SecurityUtil.utxos2Bytes(inUtxos, outUtxos);
        byte[] sendSign = SecurityUtil.signature(data, accountA.getPrivateKey());
        PublicKey sendPublicKey = accountA.getPublicKey();
        long timestamp = System.currentTimeMillis();

        transaction = new Transaction(inUtxos,outUtxos, sendSign, sendPublicKey, timestamp);

        return transaction;
    }
}

