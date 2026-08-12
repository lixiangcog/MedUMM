from medumm.core.distributed import DistributedContext


def test_distributed_context_reads_slurm_and_shards(monkeypatch):
    monkeypatch.setenv("SLURM_PROCID", "1")
    monkeypatch.setenv("SLURM_LOCALID", "1")
    monkeypatch.setenv("SLURM_NTASKS", "3")
    context = DistributedContext.from_environment()
    assert context.enabled
    assert not context.is_main_process
    assert context.shard(list(range(8))) == [1, 4, 7]
