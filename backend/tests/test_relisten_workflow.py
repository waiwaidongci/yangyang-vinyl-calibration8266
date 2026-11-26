"""返听名额锁定、IPW 保序拟合与开班工作台的领域测试。

直接运行：
    cd backend && python -m unittest discover -s tests -v
"""

import unittest
from random import Random

from src.domain.relisten_sampling import Clip, lock_review_list
from src.domain.relisten_workbench import RelistenWorkbench
from src.domain.weighted_isotonic import (
    fit_weighted_isotonic,
    identity_curve,
)


def make_clips(pairs):
    return [Clip(clip_id=cid, raw_score=score, probability=prob)
            for cid, score, prob in pairs]


class LockReviewListTests(unittest.TestCase):
    def test_guaranteed_slots_follow_calibrated_probability_descending(self):
        clips = make_clips([
            ('c1', 0.10, 0.20), ('c2', 0.50, 0.90), ('c3', 0.40, 0.70),
            ('c4', 0.30, 0.60), ('c5', 0.20, 0.40),
        ])
        locked = lock_review_list('B1', clips, quota=3, rng=Random(7), locked_at='t1')

        guaranteed = [c for c in locked.clips if c.channel == 'guarantee']
        self.assertEqual([c.clip_id for c in guaranteed], ['c2', 'c3'])
        self.assertTrue(all(c.inclusion_probability == 1.0 for c in guaranteed))
        audit = [c for c in locked.clips if c.channel == 'audit']
        self.assertEqual(len(audit), 1)
        self.assertIn(audit[0].clip_id, {'c1', 'c4', 'c5'})
        # 剩余片段为概率最低的三条（含同分排序后的剩余池），等概率抽查。
        self.assertAlmostEqual(audit[0].inclusion_probability, 1 / 3)
        self.assertEqual(locked.ranked_ids, ('c2', 'c3', 'c4', 'c5', 'c1'))
        self.assertEqual(locked.locked_at, 't1')

    def test_ties_break_by_clip_id_and_remain_deterministic(self):
        clips = make_clips([
            ('b', 0.5, 0.8), ('a', 0.4, 0.8), ('c', 0.3, 0.8), ('d', 0.2, 0.2),
        ])
        first = lock_review_list('B1', clips, 2, Random(1), 't')
        second = lock_review_list('B1', clips, 2, Random(2), 't')
        self.assertEqual(
            [c.clip_id for c in first.clips if c.channel == 'guarantee'], ['a']
        )
        self.assertEqual(first.ranked_ids, second.ranked_ids)

    def test_quota_one_is_pure_uniform_audit(self):
        clips = make_clips([('a', 0.9, 0.9), ('b', 0.1, 0.1), ('c', 0.5, 0.5)])
        locked = lock_review_list('B1', clips, 1, Random(3), 't')
        self.assertEqual(len(locked.clips), 1)
        audit = locked.clips[0]
        self.assertEqual(audit.channel, 'audit')
        self.assertAlmostEqual(audit.inclusion_probability, 1 / 3)

    def test_zero_quota_and_empty_batch(self):
        locked = lock_review_list('B1', make_clips([('a', 0.5, 0.5)]), 0, Random(1), 't')
        self.assertEqual(locked.clips, ())
        locked_empty = lock_review_list('B1', [], 0, Random(1), 't')
        self.assertEqual(locked_empty.clips, ())

    def test_quota_larger_than_population_rejected(self):
        with self.assertRaises(ValueError):
            lock_review_list('B1', make_clips([('a', 0.5, 0.5)]), 2, Random(1), 't')

    def test_audit_uniformity_across_repeated_draws(self):
        clips = make_clips([('g', 0.9, 0.95), ('a', 0.1, 0.1),
                            ('b', 0.2, 0.2), ('c', 0.3, 0.3)])
        counts = {cid: 0 for cid in ('a', 'b', 'c')}
        rng = Random(261126)
        for _ in range(6000):
            locked = lock_review_list('B1', clips, 2, rng, 't')
            audit = next(c for c in locked.clips if c.channel == 'audit')
            counts[audit.clip_id] += 1
        for hit_count in counts.values():
            self.assertAlmostEqual(hit_count / 6000, 1 / 3, delta=0.03)


class WeightedIsotonicTests(unittest.TestCase):
    def test_insufficient_samples_fall_back_to_baseline(self):
        curve = fit_weighted_isotonic([(0.2, 1, 1.0)], min_samples=8)
        self.assertEqual(curve.source, 'baseline')
        self.assertEqual(curve, identity_curve())
        self.assertAlmostEqual(curve.predict(0.37), 0.37)

    def test_inverse_probability_weights_recover_population_rate(self):
        # 总体：高分 8/10 返听，低分 2/10 返听；观察样本按设计入选。
        # 低分只以 0.25 概率入样，入样后用 1/π=4 的权重还原总体。
        samples = []
        for _ in range(8):
            samples.append((0.8, 1, 1.0))
        for _ in range(2):
            samples.append((0.8, 0, 1.0))
        for _ in range(2):
            samples.append((0.2, 1, 0.25))
        for _ in range(8):
            samples.append((0.2, 0, 0.25))
        curve = fit_weighted_isotonic(samples, min_samples=8)
        self.assertEqual(curve.source, 'fitted')
        self.assertAlmostEqual(curve.predict(0.8), 0.8, places=10)
        self.assertAlmostEqual(curve.predict(0.2), 0.2, places=10)

    def test_pava_pools_violating_blocks_and_stays_monotone(self):
        # 低分组经验率反而高，PAVA 必须把两组合并为同一平台。
        samples = [
            (0.1, 1, 1.0), (0.1, 1, 1.0),
            (0.9, 0, 1.0), (0.9, 1, 1.0),
            (0.5, 0, 1.0), (0.5, 0, 1.0),
            (0.3, 0, 1.0), (0.3, 1, 1.0),
        ]
        curve = fit_weighted_isotonic(samples, min_samples=4)
        probs = curve.probabilities
        self.assertTrue(all(left <= right for left, right in zip(probs, probs[1:])))
        self.assertEqual(len(curve.scores), len(set(curve.scores)))

    def test_invalid_inclusion_probability_rejected(self):
        with self.assertRaises(ValueError):
            fit_weighted_isotonic([(0.1, 1, 0.0)] * 8, min_samples=8)

    def test_single_label_cannot_fit(self):
        with self.assertRaises(ValueError):
            fit_weighted_isotonic([(i / 10, 1, 1.0) for i in range(1, 9)], min_samples=4)


class WorkbenchTests(unittest.TestCase):
    def setUp(self):
        self.workbench = RelistenWorkbench(min_training_samples=8, seed=261126)

    def _open_and_lock(self, batch_id, clips, quota, at):
        self.workbench.open_batch(batch_id, at=f'{at}-open')
        return self.workbench.lock_batch_list(batch_id, clips, quota, at=f'{at}-lock')

    def _complete_batch(self, batch_id, labels, at):
        locked = self.workbench._batches[batch_id].locked_list
        for clip, label in zip(locked.clips, labels):
            self.workbench.record_conclusion(batch_id, clip.clip_id, label, at=at)

    def test_first_batch_uses_baseline_curve(self):
        info = self.workbench.open_batch('B1', at='t0')
        self.assertEqual(info['curve_source'], 'baseline')
        self.assertEqual(info['training_batches'], ())
        self.assertEqual(info['training_samples'], 0)

    def test_retry_returns_identical_locked_list_and_probabilities(self):
        clips = [{'id': f'c{i}', 'raw_score': i / 10} for i in range(1, 9)]
        self._open_and_lock('B1', clips, quota=3, at='t1')
        first = self.workbench.export_batch('B1')

        retry = self.workbench.lock_batch_list('B1', clips, quota=3, at='t2')
        self.assertEqual(retry.locked_at, 't1-lock')
        second = self.workbench.export_batch('B1')
        self.assertEqual(first, second)
        self.assertEqual(
            [row['inclusion_probability'] for row in first],
            [row['inclusion_probability'] for row in second],
        )

    def test_consecutive_complete_batches_feed_weighted_refit(self):
        clips_a = [{'id': f'a{i}', 'raw_score': i / 10} for i in range(1, 9)]
        locked_a = self._open_and_lock('B1', clips_a, 4, 't1')
        self.assertEqual(len(locked_a.clips), 4)
        labels_a = [1, 1, 0, 0]
        self._complete_batch('B1', labels_a, 't2')

        info_b = self.workbench.open_batch('B2', at='t3')
        self.assertEqual(info_b['training_batches'], ('B1',))
        self.assertEqual(info_b['training_samples'], 4)
        # 样本不足阈值，沿用基准曲线而不是强行拟合。
        self.assertEqual(info_b['curve_source'], 'baseline')

    def test_prefix_breaks_on_incomplete_batch(self):
        clips = [{'id': f'x{i}', 'raw_score': i / 10} for i in range(1, 9)]
        self._open_and_lock('B1', clips, 4, 't1')
        self._complete_batch('B1', [1, 0, 1, 0], 't2')

        clips_b = [dict(row, id=f'b{i}') for i, row in enumerate(clips)]
        locked_b = self._open_and_lock('B2', clips_b, 4, 't3')
        # B2 只补录一半，未收齐。
        for clip in list(locked_b.clips)[:2]:
            self.workbench.record_conclusion('B2', clip.clip_id, 1, at='t4')

        self.workbench.open_batch('B3', at='t5')
        clips_c = [dict(row, id=f'c{i}') for i, row in enumerate(clips)]
        self.workbench.lock_batch_list('B3', clips_c, 4, at='t6')
        self._complete_batch('B3', [0, 0, 1, 1], 't7')

        info_d = self.workbench.open_batch('B4', at='t8')
        # B2 截断连续前缀；B3 虽然收齐也不能进入训练。
        self.assertEqual(info_d['training_batches'], ('B1',))

    def test_late_backfill_uses_lock_timepoint_and_never_changes_list(self):
        clips = [{'id': f'c{i}', 'raw_score': i / 10} for i in range(1, 9)]
        locked = self._open_and_lock('B1', clips, 3, 't1')
        before = self.workbench.export_batch('B1')

        # 模拟乱序补录：先登记最后一条，再登记第一条。
        ordered = list(locked.clips)
        self.workbench.record_conclusion('B1', ordered[-1].clip_id, 0, at='t3')
        self.workbench.record_conclusion('B1', ordered[0].clip_id, 1, at='t4')
        self.workbench.record_conclusion('B1', ordered[1].clip_id, 1, at='t5')

        after = self.workbench.export_batch('B1')
        self.assertEqual(
            [row['inclusion_probability'] for row in before],
            [row['inclusion_probability'] for row in after],
        )
        self.assertEqual([row['id'] for row in before], [row['id'] for row in after])
        labels = {row['id']: row['needs_relisten'] for row in after}
        self.assertEqual(labels[ordered[-1].clip_id], 0)
        self.assertEqual(labels[ordered[0].clip_id], 1)

        with self.assertRaises(ValueError):
            self.workbench.record_conclusion('B1', ordered[0].clip_id, 0, at='t6')
        with self.assertRaises(ValueError):
            self.workbench.record_conclusion('B1', 'not-in-list', 1, at='t6')

    def test_concurrent_open_reads_conclusions_visible_at_open_time(self):
        clips = [{'id': f'c{i}', 'raw_score': i / 10} for i in range(1, 9)]
        self._open_and_lock('B1', clips, 4, 't1')
        self._complete_batch('B1', [1, 1, 1, 1], 't2')

        # B2 在 t3 开班时 B1 已收齐，训练集包含 B1；之后补录不追溯。
        info = self.workbench.open_batch('B2', at='t3')
        self.assertEqual(info['training_batches'], ('B1',))
        self.assertEqual(info['training_samples'], 4)

    def test_export_carries_locked_inclusion_probability_and_snapshot_roundtrip(self):
        clips = [{'id': f'c{i}', 'raw_score': i / 10} for i in range(1, 9)]
        self._open_and_lock('B1', clips, 3, 't1')
        rows = self.workbench.export_batch('B1')
        self.assertEqual({row['selection_channel'] for row in rows},
                         {'guarantee', 'audit'})
        self.assertTrue(all(0 < row['inclusion_probability'] <= 1 for row in rows))
        guaranteed_p = [row['inclusion_probability'] for row in rows
                        if row['selection_channel'] == 'guarantee']
        self.assertTrue(all(p == 1.0 for p in guaranteed_p))
        self.assertTrue(all(row['needs_relisten'] is None for row in rows))

        snapshot = self.workbench.to_snapshot()
        restored = RelistenWorkbench.from_snapshot(snapshot)
        self.assertEqual(restored.export_batch('B1'), rows)
        # 快照恢复后重试仍是同一份名单。
        self.assertEqual(
            restored.lock_batch_list('B1', clips, 3).clip_ids,
            self.workbench.lock_batch_list('B1', clips, 3).clip_ids,
        )

    def test_duplicate_open_and_unknown_batch_rejected(self):
        self.workbench.open_batch('B1')
        with self.assertRaises(ValueError):
            self.workbench.open_batch('B1')
        with self.assertRaises(ValueError):
            self.workbench.curve_for('nope')


if __name__ == '__main__':
    unittest.main()
