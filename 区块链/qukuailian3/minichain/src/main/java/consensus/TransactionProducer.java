package consensus;

import data.*;
import utils.SecurityUtil;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.UUID;

/**
 * 生成随机交易
 */
public class TransactionProducer extends Thread {

    private TransactionPool transactionPool;
    private BlockChain blockChain;

    public TransactionProducer(TransactionPool transactionPool,BlockChain blockChain) {

        this.transactionPool = transactionPool;
        this.blockChain=blockChain;
    }

    @Override
    public void run() {
        while (true) {
            synchronized (transactionPool) {
                while (transactionPool.isFull()) {
                    try {
                        transactionPool.wait();
                    } catch (InterruptedException e) {
                        e.printStackTrace();
                    }
                }
                Transaction randomOne = getOneTransaction();
                transactionPool.put(randomOne);
                if (transactionPool.isFull()) {
                    transactionPool.notify();
                }
            }
        }
    }

    private Transaction getOneTransaction() {
//        Transaction transaction = new Transaction(UUID.randomUUID().toString(), System.currentTimeMillis());
//        return transaction;
        Random random = new Random(); // random.nextInt(bound) 在[0， bound) 中取
        Transaction transaction = null; // 返回的交易
        Account[] accounts = blockChain.getAccounts(); // 获取账户数组
        while (true) {
            // 随机获取两个账户A和B
            Account aAccount = accounts[random.nextInt(accounts.length)];
            Account bAccount = accounts[random.nextInt(accounts.length)];
            // BTC不允许自己给自己转账
            if (aAccount == bAccount) {
                continue;
            }
            // 获得钱包地址
            String aWalletAddress = aAccount.getWalletAddress();
            String bWalletAddress = bAccount.getWalletAddress();
            // 获取A可用的Utxo并计算余额
            UTXO[] aTrueUtxos = blockChain.getTrueUtxos(aWalletAddress);
            int aAmount = aAccount.getAmount(aTrueUtxos);
            //如果A账户的余额为0，则无法构建交易，重新随机生成
            if (aAmount == 0) {
                continue;
            }
            // 随机生成交易数额 [1，aAmount] 之间
            int txAmount = random.nextInt(aAmount) + 1;
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
            if (inAmount < txAmount) {
                continue;
            }
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
            break;
        }
        return transaction;
    }

}
