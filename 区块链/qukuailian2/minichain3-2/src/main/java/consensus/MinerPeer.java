package consensus;

import config.MiniChainConfig;
import data.*;
import network.Network;
import spv.Proof;
import spv.SpvPeer;
import utils.MinerUtil;
import utils.SecurityUtil;

import java.security.PublicKey;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * 矿工线程
 *
 * 该线程的主要工作就是不断的进行交易打包、Merkle树根哈希值计算、构造区块，
 * 然后尝试使用不同的随机字段（nonce）进行区块的哈希值计算以生成新的区块添加到区块中
 *
 * 这里需要你实现的功能函数为：getBlockBody、getMerkleRootHash、mine和getBlock，具体的需求见上述方法前的注释，
 * 除此之外，该类中的其他方法、变量，以及其他类中的方法和变量，均无需修改，否则可能影响系统的正确运行
 *
 * 如有疑问，及时交流
 *
 */
public class MinerPeer extends Thread {

    private final BlockChain blockChain;
    private final Network network;

    public MinerPeer(BlockChain blockChain, Network network) {
        this.blockChain = blockChain;
        this.network = network;
    }

    @Override
    public void run() {
        while (true) {
            synchronized (network.getTransactionPool()) {
                TransactionPool transactionPool = network.getTransactionPool();

                while (!transactionPool.isFull()) {
                    try {
                        transactionPool.wait();
                    } catch (InterruptedException e) {
                        e.printStackTrace();
                    }
                }

                // 从交易池中获取一批次的交易
                Transaction[] transactions = transactionPool.getAll();
                //对交易的签名进行验签，失败则退出
                if(!check(transactions)){
                    System.out.println("transactions error");
                    System.exit(-1);
                }
                // 以交易为参数，调用getBlockBody方法
                BlockBody blockBody = getBlockBody(transactions);

                // 以blockBody为参数，调用mine方法
                Block block = mine(blockBody);
                // 将挖出的block广播到网络
                boardcast(block);


                //输出所有账户余额总数
                System.out.println("the sum of all account: "+blockChain.getAllAccountAmount());

                transactionPool.notify();
            }
        }
    }
    private boolean check(Transaction[] transactions){
        for(int i=0;i<transactions.length;i++){
            Transaction transaction=transactions[i];
            //签名的数据是该交易的inUtxos和outUtxos
            byte[] data= SecurityUtil.utxos2Bytes(transaction.getInUtxos(),transaction.getOutUtxos());
            byte[] sign=transaction.getSendSign();
            PublicKey publicKey=transaction.getSendPublicKey();
            if(!SecurityUtil.verify(data,sign,publicKey)){
                return false;
            }
        }
        return true;
    }

    /**
     * 该方法根据传入的参数中的交易，构造并返回一个相应的区块体对象
     *
     * 查看BlockBody类中的字段以及构造方法你会发现，还需要根据这些交易计算Merkle树的根哈希值
     *
     * @param transactions 一批次的交易
     *
     * @return 根据参数中的交易构造出的区块体
     */
    public BlockBody getBlockBody(Transaction[] transactions) {
        assert transactions != null && transactions.length == MiniChainConfig.MAX_TRANSACTION_COUNT;
        //
        List<String> list = new ArrayList<>();
        for (Transaction transaction: transactions) {
            String txHash = SecurityUtil.sha256Digest(transaction.toString());
            list.add(txHash);
        }
        // list大小为1时停止迭代
        while (list.size() != 1) {
            List<String> newList = new ArrayList<>();
            for (int i = 0; i < list.size(); i += 2) {
                String leftHash = list.get(i);
                // 如果出现奇数个节点，即最后一个节点没有右结点与其构成一对，就将当前节点复制一份作为右节点
                String rightHash = (i + 1 < list.size() ? list.get(i + 1) : leftHash);
                String parentHash = SecurityUtil.sha256Digest(leftHash + rightHash);
                newList.add(parentHash);
            }
            // 切换list，进行下一轮的计算
            list = newList;
        }
        BlockBody blockBody = new BlockBody(list.get(0), transactions);
        return blockBody;
    }

    /**
     * 该方法即在循环中完成"挖矿"操作，其实就是通过不断的变换区块中的nonce字段，直至区块的哈希值满足难度条件，
     * 即可将该区块加入区块链中
     *
     * @param blockBody 区块体
     * @return
     */
    private Block mine(BlockBody blockBody) {
        Block block = getBlock(blockBody);
        while (true) {
            String blockHash = SecurityUtil.sha256Digest(block.toString());
            if (blockHash.startsWith(MinerUtil.hashPrefixTarget())) {
                System.out.println("Mined a new Block! Detail of the new Block : ");
                System.out.println(block.toString());
                System.out.println("And the hash of this Block is : " + SecurityUtil.sha256Digest(block.toString()) +
                                    ", you will see the hash value in next Block's preBlockHash field.");
                System.out.println();
                blockChain.addNewBlock(block);
                break;
            } else {
                //todo
                Random random = new Random();
                long nonce = random.nextLong();
                BlockHeader blockHeader=block.getBlockHeader();
                blockHeader.setNonce(nonce);

            }
        }
        return block;
    }

    /**
     * 该方法供mine方法调用，其功能为根据传入的区块体参数，构造一个区块对象返回，
     * 也就是说，你需要构造一个区块头对象，然后用一个区块对象组合区块头和区块体
     *
     * 建议查看BlockHeader类中的字段和注释，有助于你实现该方法
     *
     * @param blockBody 区块体
     *
     * @return 相应的区块对象
     */
    public Block getBlock(BlockBody blockBody) {
        //todo
        //构造区块头
        Block preblock=blockChain.getNewestBlock();
        String preBlockHash=SecurityUtil.sha256Digest(preblock.toString());
        String merkleRootHash=blockBody.getMerkleRootHash();
        Random random = new Random();

        // 生成随机的long值
        long nonce = random.nextLong();
        BlockHeader blockHeader=new BlockHeader(preBlockHash, merkleRootHash, nonce);
        //构造并创建实例
        return new Block(blockHeader,blockBody);
    }
    public Proof getProof(String proofTxHash){
        Block proofBlock = null;
        int proofHeight =-1;
        //遍历链上所有区块内的所有交易，计算其哈希，找出要验证哈希值的交易所在的区块
//        System.out.println("proofTxHash=="+proofTxHash);
        for(Block block:blockChain.getBlocks()){
            ++proofHeight;//获得区块高度(我们这里从0开始索引)
            for(Transaction transaction: block.getBlockBody().getTransactions()){

                String txHash = SecurityUtil.sha256Digest(transaction.toString());
//                System.out.println("#############\n");
//                System.out.println(txHash);
//                System.out.println("#############\n");
                if(txHash.equals(proofTxHash)){
                    proofBlock = block;
                    break;
                }
            }
            if (proofBlock!=null){
                break;
            }
        }
        if (proofBlock==null){
            return null;
        }
        //重新计算merkle树获得路径哈希值，同时记录相关的节点偏向信息，构建验证路径节点
        List<Proof.Node> proofPath =new ArrayList<>();
        List<String> list = new ArrayList<>();
        String pathHash = proofTxHash; // 路径哈希，即验证节点一路延申至根节点的哈希值
        for (Transaction transaction: proofBlock.getBlockBody().getTransactions()){
            String txHash = SecurityUtil.sha256Digest(transaction.toString());
            list.add(txHash);
        }
        // list大小为1时停止迭代
        while (list.size()!=1){
            List<String>newList =new ArrayList<>();
            for(int i=0;i<list.size();i+=2){
                String leftHash = list.get(i);
                //如果出现奇数个节点，即最后一个节点没有右结点与其构成一对，就将当前节点复制一份作为右节点
                String rightHash =(i + 1<list.size()? list.get(i + 1): leftHash);
                String parentHash = SecurityUtil.sha256Digest( leftHash + rightHash);
                newList.add(parentHash);
                //如果某一个哈希值与路径哈希相同，则将另一个作为验证路径中的节点加入，同时记录偏向,并更新路径哈希
                if (pathHash.equals(leftHash)){
                    Proof.Node proofNode = new Proof.Node(rightHash, Proof.Orientation.RIGHT);
                    proofPath.add(proofNode);
                    pathHash = parentHash;
                }else if (pathHash.equals(rightHash)){
                    Proof.Node proofNode = new Proof.Node(leftHash, Proof.Orientation.LEFT);
                    proofPath.add(proofNode);
                    pathHash = parentHash;
                }
            }
            // 切换list，进行下一轮的计算
            list = newList;
        }
        String proofMerkleRootHash =list.get(0);
        // 构造Proof并返回
        return new Proof(proofTxHash, proofMerkleRootHash, proofHeight, proofPath);

    }

    public void boardcast(Block block){
        // 每个spv节点接受区块头
        SpvPeer spvPeer=network.getSpvPeer();
        spvPeer.accept(block.getBlockHeader());
    }
}
