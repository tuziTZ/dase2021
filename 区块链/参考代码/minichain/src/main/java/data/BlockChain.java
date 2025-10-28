package data;

import config.MiniChainConfig;
import utils.SecurityUtil;

import java.nio.charset.StandardCharsets;
import java.security.KeyPair;
import java.security.PrivateKey;
import java.security.PublicKey;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.Random;
import java.util.Set;

/**
 * 区块链的类抽象，创建该对象时会自动生成创世纪块，加入区块链中
 */
public class BlockChain {

    private final LinkedList<Block> chain = new LinkedList<>();
    private final Account[] accounts;

    public BlockChain() {
        this.accounts = new Account[MiniChainConfig.ACCOUNT_NUM];
        for (int i = 0; i < accounts.length; i++) {
            accounts[i] = new Account();
        }
        // 在创世区块中为每个账户分配一定金额的utxo，比那与后面交易的进行
        Transaction[] transactions = genesisTransactions(accounts);

        BlockHeader genesisBlockHeader = new BlockHeader(null, null,
                Math.abs(new Random().nextLong()));

        BlockBody genesisBlockBody = new BlockBody(null, transactions);
        Block genesisBlock = new Block(genesisBlockHeader, genesisBlockBody);
        System.out.println("Create the genesis Block! ");
        System.out.println("And the hash of genesis Block is : " + SecurityUtil.sha256Digest(genesisBlock.toString()) +
                ", you will see the hash value in next Block's preBlockHash field.");
        System.out.println();
        chain.add(genesisBlock);
    }

    private Transaction[] genesisTransactions(Account[] accounts) {
        UTXO[] outUtxos = new UTXO[accounts.length];
        for (int i = 0; i < accounts.length; i++) {
            outUtxos[i] = new UTXO(accounts[i].getWalletAddress(), MiniChainConfig.INIT_ACCOUNT, accounts[i].getPublicKey());
        }

        KeyPair dayDreamKeyPair = SecurityUtil.secp256k1Generate();
        PublicKey dayDreamPublicKey = dayDreamKeyPair.getPublic();
        PrivateKey dayDreamPrivateKey = dayDreamKeyPair.getPrivate();
        // 签名
        byte[] sign = SecurityUtil.signature("Everything in the dream!".getBytes(StandardCharsets.UTF_8), dayDreamPrivateKey);
        return new Transaction[]{new Transaction(new UTXO[]{}, outUtxos, sign, dayDreamPublicKey, System.currentTimeMillis())};
    }

    /**
     * 向区块链中添加新的满足难度条件的区块
     *
     * @param block 新的满足难度条件的区块
     */
    public void addNewBlock(Block block) {
        chain.offer(block);
    }

    /**
     * 获取区块链的最后一个区块，矿工在组装新的区块时，需要获取上一个区块的哈希值，通过该方法获得
     *
     * @return 区块链的最后一个区块
     */
    public Block getNewestBlock() {
        return chain.peekLast();
    }

    /**
     * 遍历整个区块链获得某钱包地址相关的utxo，获得真正的utxo，即未被使用的utxo
     * @param walletAddress 钱包地址
     * @return
     */
    public UTXO[] getTrueUtxos(String walletAddress) {
        // 使用哈希表存储结果，保证每个utxo唯一
        Set<UTXO> trueUtxoSet = new HashSet<>();
        // 遍历每个区块
        for (Block block : chain) {
            BlockBody blockBody = block.getBlockBody();
            Transaction[] transactions = blockBody.getTransactions();
            // 遍历区块中的所有交易
            for (Transaction transaction : transactions) {
                UTXO[] inUtxos = transaction.getInUtxos();
                UTXO[] outUtxos = transaction.getOutUtxos();
                // 交易中的inUtxo是已使用的utxo，故需要删除
                for (UTXO utxo : inUtxos) {
                    if (utxo.getWalletAddress().equals(walletAddress)) {
                        trueUtxoSet.remove(utxo);
                    }
                }
                // 交易中的outUtxo是新产生的utxo，可作为后续交易使用
                for (UTXO utxo : outUtxos) {
                    if (utxo.getWalletAddress().equals(walletAddress)) {
                        trueUtxoSet.add(utxo);
                    }
                }
            }
        }
        // 转化为数组形式返回
        UTXO[] trueUtxos = new UTXO[trueUtxoSet.size()];
        trueUtxoSet.toArray(trueUtxos);
        return trueUtxos;
    }

    public Account[] getAccounts() {
        return accounts;
    }

    public int getAllAccountAmount() {
        int sumAmount = 0;
        for (int i = 0; i < accounts.length; ++i) {
            UTXO[] trueUtxo = getTrueUtxos(accounts[i].getWalletAddress());
            sumAmount += accounts[i].getAmount(trueUtxo);
        }
        return sumAmount;
    }
}
