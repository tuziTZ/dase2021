package spv;

import consensus.MinerPeer;
import data.*;
import network.Network;
import utils.SecurityUtil;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.Scanner;

public class SpvPeer extends Thread {
    // 该spv节点只存储区块头
    private final List<BlockHeader> headers= new ArrayList<>();
    // 该spV拥有一个账户信息
    private Account account;
    // 节点连接到网络
    private final Network network;
    public SpvPeer(Account account,Network network){
        this.account = account;
        this.network = network;
    }
    @Override
    public void run() {
        while (true) {
            synchronized (network.getTransactionPool()) {
                TransactionPool transactionPool = network.getTransactionPool();

                while (transactionPool.isFull()) {
                    try {
                        transactionPool.wait();
                    } catch (InterruptedException e) {
                        e.printStackTrace();
                    }
                }
                Scanner scan=new Scanner(System.in);
                if(account==null)//先创建用户才能进入功能交互界面
                {
                    System.out.println("You should create a account! Pease enter:create a account");
                    String str=scan.nextLine();
                    if(str.equals("create a account"))
                    {
                        account=network.create_account();
                        System.out.println("create successfully! account is "+account.getWalletAddress());

                    }else{
                        System.out.println("You enter wrong.");
                    }
                }else{
                    //功能说明
                    System.out.println("Now you can choose the following features:");
                    System.out.println("create your own transaction(enter:create a transaction).");
                    System.out.println("query your balance(enter:query balance).");
                    System.out.println("query a transaction(enter:query transaction).");
                    System.out.println("create random transactions(enter:random transaction).");
                }
                String str=scan.nextLine();
                System.out.println(str+":");
                if(str.equals("")){
                    continue;
                }

                if(str.equals("create a transaction"))//创建转账交易
                {
                    System.out.println("please enter the index of the account you want to transfer and the number of money.");
                    int toaccount_index=scan.nextInt();
                    int amount=scan.nextInt();
                    Account toaccount=network.getAccounts().get(toaccount_index);
                    Transaction transaction =getOneTransaction(toaccount,amount);
                    if(transaction==null)
                        continue;
                    System.out.println("create a transaction, the txHash is "+SecurityUtil.sha256Digest(transaction.toString()));
                    transactionPool.put(transaction);
                    if(transactionPool.isFull()) {
                        transactionPool.notify();
                    }
                }
                else if(str.equals("query balance")){
                    int amount = getbalance();
                    System.out.println("The balance of your account is "+amount);

                }
                else if (str.equals("query transaction")){
                    System.out.println("please enter a txHash of transaction.");
                    String txHash=scan.nextLine();
                    if(simplifiedPaymentVerify(txHash)){
                        System.out.println("transaction exist.");}
                    else {
                        System.out.println("transaction doesn't exist.");
                    }
                }
                else if (str.equals("random transaction")){
                    while(!transactionPool.isFull()){
                        Transaction transaction=getRandomTransaction();
                        System.out.println("create random transantion, the txHash is "+SecurityUtil.sha256Digest(transaction.toString()));
                        transactionPool.put(transaction);
                        if(transactionPool.isFull()){
                            transactionPool.notify();
                            break;
                        }
                    }

                }
                System.out.println();

            }
        }
    }

    private int getbalance() {
        String walletAddress=account.getWalletAddress();
        UTXO[] aTrueUtxos = network.getBlockChain().getTrueUtxos(walletAddress);
        int amount = account.getAmount(aTrueUtxos);
        return amount;
    }

    private Transaction getOneTransaction(Account toaccount, int amount){
        Transaction transaction = null;// 返回的交易
        List<Account> accounts = network.getAccounts(); // 从网络中获取账户数组
        Account aAccount = account;
        Account bAccount = toaccount;
        // BTC不允许自己给自己转账
        if(aAccount == bAccount){
            System.out.println("you can't transfer to yourself.");
            return transaction;
        }
        String aWalletAddress = aAccount.getWalletAddress();
        String bWalletAddress = bAccount.getWalletAddress();
        // 获取A可用的utxo并计算余额
        UTXO[] aTrueUtxos = network.getBlockChain().getTrueUtxos(aWalletAddress);
        int aAmount = getbalance();
        int txAmount = amount;
// 如果A账户的余额小于amount，则无法构建交易。
        if(aAmount <amount) {
            System.out.println("your balance can't afford this transaction. your balance is " + aAmount);
            return transaction;
        }
        // 构建Inutxo和oututxo
        List<UTXO>inUtxoList = new ArrayList<>();
        List<UTXO>outUtxoList = new ArrayList<>();
        //A账户需先解锁才能使用自己的utxo，解锁需要私钥签名和公钥去执行解锁脚本，这里先生成需要解锁的签名
        //签名的数据我们约定为公钥的二进制数据
        byte[] aUnlockSign = SecurityUtil.signature(aAccount.getPublicKey().getEncoded(), aAccount.getPrivateKey());
        //选择输入总额>=交易数额的 utxo
        int inAmount =0;
        for(UTXO utxo :aTrueUtxos){
// 解锁成功才能使用该utxo
            if(utxo.unlockScript(aUnlockSign, aAccount.getPublicKey())){
                inAmount += utxo.getAmount();
                inUtxoList.add(utxo);
                if(inAmount >= txAmount){
                    break;
                }
            }
        }
        //可解锁的utxo总额仍不足以支付交易数额，则重新随机
        if(inAmount<txAmount){
            System.out.println("the unlocked utxos is not enough.");
            return transaction;
        }
        //构建输出oututxos，A账户向B账户支付txAmount，同时输入对方的公钥以供生成公钥哈希
        outUtxoList.add(new UTXO(bWalletAddress, txAmount, bAccount .getPublicKey()));
        //如果有余额，则“找零”，即给自己的utxo
        if(inAmount >txAmount){
            outUtxoList.add(new UTXO(aWalletAddress, inAmount - txAmount, aAccount.getPublicKey()));
        }
        //导出固定utxo数组
        UTXO[] inUtxos = inUtxoList.toArray(new UTXO[0]);
        UTXO[] outUtxos = outUtxoList.toArray(new UTXO[0]);
        // A账户需对整个交易进行私钥签名，确保交易不会被篡改，因为交易会传输到网络中，而上述步骤可在本地离线环境中构造
        // 获取要签名的数据，这个数据需要囊括交易信息
        byte[] data=SecurityUtil.utxos2Bytes(inUtxos, outUtxos);
        // A账户使用私钥签名
        byte[] sign = SecurityUtil.signature(data, aAccount.getPrivateKey());
        // 交易时间戳
        long timestamp=System.currentTimeMillis();
        // 构造交易
        transaction = new Transaction(inUtxos, outUtxos, sign, aAccount.getPublicKey(), timestamp);
        return transaction;

    }
    private Transaction getRandomTransaction() {
//        Transaction transaction = new Transaction(UUID.randomUUID().toString(), System.currentTimeMillis());
//        return transaction;
        Random random = new Random(); // random.nextInt(bound) 在[0， bound) 中取
        Transaction transaction = null; // 返回的交易
        Account[] accounts = network.getAccounts().toArray(new Account[0]); // 获取账户数组
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
            UTXO[] aTrueUtxos = network.getBlockChain().getTrueUtxos(aWalletAddress);
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
    public void accept(BlockHeader blockHeader){
        headers.add(blockHeader);
        verifyLatest();
    }
    //*如果有相关的交易，验证最新块的交易
    public void verifyLatest(){
        if (account==null){
            return;
        }
        // spv节点通过网络搜集最新块与自己相关的交易
        List<Transaction> transactions = network.getTransactionsInLatestBlock(account.getWalletAddress());
        if(transactions.isEmpty()){
            return;
        }
        // 富翁使用自己"贫瘠不堪"的spv节点使用spv请求验证他参与的交易
        System.out.println("Account[" + account.getWalletAddress()+ "] began to verify the transaction...");
        for (Transaction transaction:transactions){
            if(!simplifiedPaymentVerify(SecurityUtil.sha256Digest(transaction.toString()))){
                // 因为理论上肯定能验证成功，如果失败说明程序出现了bug，所以直接退出
                System.out.println("verification failed!");
                System.exit( -1);
            }
        }
        System.out.println("Account[" + account.getWalletAddress() + "] verifies all transactions are successful! \n");
    }

    public boolean simplifiedPaymentVerify(String txHash){
        // 获取交易哈希
        // 通过网络向其他全节点获取验证路径(我们这里网络只有矿工一个全节点)
        MinerPeer minerPeer =network.getMinerPeer();
        Proof proof = minerPeer.getProof(txHash);
        if(proof == null){
            return false;
        }
        //使用获得的验证路径计算merkel根哈希
        String hash = proof.getTxHash();
        for(Proof.Node node: proof.getPath()){
            //此处可査看 MinerPeer 节点类里计算根哈希的方式
            switch (node.getOrientation()){
                case LEFT: hash = SecurityUtil.sha256Digest( node.getTxHash()+ hash); break;
                case RIGHT: hash = SecurityUtil.sha256Digest( hash + node.getTxHash()); break;
                default: return false;
            }
        }
        // 获得本地区块头部中的根哈希
        int height = proof.getHeight();
        String localMerkleRootHash = headers.get(height).getMerkleRootHash();
        //获取远程节点发送过来的根哈希
        String remoteMerkleRootHash = proof.getMerkleRootHash();
        // 调试
        System.out.println("\n--------> verify hash:\t" + txHash);
        System.out.println("calMerkleRootHash:\t\t" + hash);
        System.out.println("localMerkleRootHash:\t" + localMerkleRootHash);
        System.out.println("remoteMerkleRootHash:\t" + remoteMerkleRootHash);
        System.out.println();
        //判断生成的根哈希与本地的根哈希和远程的根哈希是否相等

        return hash.equals(localMerkleRootHash)&& hash.equals(remoteMerkleRootHash);
    }
}
