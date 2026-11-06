// 批量识别结果项详情弹窗（2026-10-30 新增）
//
// 为什么批量结果需要单独一个弹窗，而不复用 RecordDetailModal：
//   RecordDetailModal 是「按 record_id 向后端要详情」，而批量结果里的每一项都是
//   刚算出来的、数据就在内存里（含本地缩略图）。等一次网络往返再展示没有任何收益，
//   还会在匿名（未登录、无 record_id）场景下直接失配 —— 那时根本没有 id 可查。
//
// 为什么把它做成「可对单张重裁」：
//   批量场景刻意不做逐张裁剪（9 张一张张框太慢），但实测里整株远景是误判第一大原因
//   （docs/23：单叶近景 Top-1 43.6%，整株远景 17%）。所以留一个入口：
//   在批量结果里挑出不准的那张，单独框选后重识别，不要整批重来。
import React from 'react'
import { Modal, Image, Typography, Tag, Button, Progress, Descriptions, Alert, Space } from 'antd'
import { ScissorOutlined, HistoryOutlined, PictureOutlined } from '@ant-design/icons'
import { formatPercent } from '../utils/format'
import { diseaseRefs } from '../utils/diseaseRefs'

// 可靠性分档文案：与识别页保持一致（同一份后端字段，两处文案不同会造成困惑）
const RELIABILITY_META = {
  high: { color: 'green', text: '结果较可靠' },
  medium: { color: 'gold', text: '结果仅供参考' },
  low: { color: 'red', text: '结果可能不可靠' },
  unknown: { color: 'default', text: '未校准' },
}

export default function BatchItemModal({ open, item, preview, onClose, onRecrop }) {
  if (!item) return null

  const isHealthy = !!item.is_healthy
  const top3 = item.top3 || []
  const knowledge = item.knowledge || {}
  const rel = RELIABILITY_META[item.reliability] || RELIABILITY_META.unknown
  // 与识别页同一条规则：「大概率不是叶片」时只提示这一件事，
  // 不再叠加「置信度低」，否则两条警告互相打架，用户不知道该信哪条
  const showLowConf = !item.likely_not_leaf && item.need_review

  return (
    <Modal
      open={open}
      onCancel={onClose}
      footer={null}
      width={860}
      title={`第 ${item.index + 1} 张 · ${item.file_name || '未命名'}`}
    >
      <div style={{ display: 'flex', gap: 20, alignItems: 'flex-start' }}>
        {/* 左：图。用 antd Image 的预览组，点开可放大 / 旋转 / 缩放 ——
            看叶片病斑时「放大」不是锦上添花，是能不能看清的关键。 */}
        <div style={{ width: 320, flexShrink: 0 }}>
          {preview ? (
            <Image
              src={preview}
              alt={item.file_name || '叶片图片'}
              style={{
                width: '100%',
                borderRadius: 8,
                border: '1px solid var(--c-border)',
                display: 'block',
              }}
            />
          ) : (
            <div
              style={{
                width: '100%',
                aspectRatio: '1 / 1',
                borderRadius: 8,
                border: '1px solid var(--c-border)',
                background: 'var(--c-bg-subtle)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                color: 'var(--c-text-placeholder)',
              }}
            >
              <PictureOutlined style={{ fontSize: 28 }} />
            </div>
          )}
          <Typography.Text type="secondary" style={{ fontSize: 12, display: 'block', marginTop: 8 }}>
            点击图片可放大查看病斑细节
          </Typography.Text>
        </div>

        {/* 右：结论 + 候选 + 建议 */}
        <div style={{ flex: 1, minWidth: 0 }}>
          <Space size={6} wrap style={{ marginBottom: 8 }}>
            <Tag color={isHealthy ? 'green' : 'volcano'}>{isHealthy ? '健康' : '病害'}</Tag>
            {item.crop_applied && (
              <Tag style={{ margin: 0 }}>已限定作物 {item.crop_candidates} 个候选</Tag>
            )}
            {item.calibration_enabled && <Tag color={rel.color}>{`校准：${rel.text}`}</Tag>}
          </Space>

          <div style={{ display: 'flex', alignItems: 'baseline', gap: 12, flexWrap: 'wrap' }}>
            <Typography.Title level={4} style={{ margin: 0 }}>
              {item.disease_cn}
            </Typography.Title>
            <span style={{ fontSize: 22, fontWeight: 600, color: 'var(--c-primary-600)' }}>
              {formatPercent(item.confidence)}
            </span>
            <Typography.Text type="secondary" style={{ fontSize: 12 }}>
              Top-1 置信度
            </Typography.Text>
          </div>
          {item.calibration_enabled && typeof item.confidence_calibrated === 'number' && (
            <Typography.Text
              style={{
                fontSize: 12,
                color:
                  item.reliability === 'low'
                    ? 'var(--c-error-600)'
                    : item.reliability === 'medium'
                    ? 'var(--c-warning-600)'
                    : 'var(--c-success-600)',
              }}
            >
              真实场景校准：{formatPercent(item.confidence_calibrated)}
            </Typography.Text>
          )}

          {item.likely_not_leaf && (
            <Alert
              type="error"
              showIcon
              style={{ marginTop: 10 }}
              message="这张可能不是叶片照片"
              description={`系统判断为「叶片」的把握仅 ${formatPercent(item.leaf_prob)}，下方结果不可信，请勿据此用药。`}
            />
          )}
          {showLowConf && (
            <Alert
              type="warning"
              showIcon
              style={{ marginTop: 10 }}
              message="置信度较低"
              description="低于 50%，建议放大核对或框选单片叶片后重新识别。"
            />
          )}
          {!item.likely_not_leaf && item.grayscale && (
            <Alert
              type="warning"
              showIcon
              style={{ marginTop: 10 }}
              message="这是黑白 / 灰度图片"
              description="病害识别依赖颜色（黄化、褐变、霉层），灰度图会削弱判断依据。"
            />
          )}

          <Typography.Paragraph strong style={{ margin: '14px 0 6px' }}>
            Top-3 可能性
            <Typography.Text type="secondary" style={{ fontSize: 12, fontWeight: 400, marginLeft: 8 }}>
              点图片看症状示例
            </Typography.Text>
          </Typography.Paragraph>
          {/* 参考图放在 PreviewGroup 里：三张可左右切换看，比一张张点开更符合
              「对照病斑形状 / 分布 / 颜色」这个真实用途 */}
          <Image.PreviewGroup>
            {top3.map((t, i) => {
              const refs = diseaseRefs(t.class_id)
              return (
                <div key={i} style={{ marginBottom: 8 }}>
                  <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
                    <div
                      style={{
                        width: 44,
                        height: 44,
                        flexShrink: 0,
                        borderRadius: 6,
                        overflow: 'hidden',
                        border: '1px solid var(--c-border)',
                        background: 'var(--c-bg-subtle)',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                      }}
                    >
                      {refs.length ? (
                        <Image
                          src={refs[0]}
                          alt={`${t.disease_cn}参考图`}
                          width={44}
                          height={44}
                          style={{ objectFit: 'cover' }}
                          preview={{ src: refs[0] }}
                        />
                      ) : (
                        <PictureOutlined style={{ color: 'var(--c-text-placeholder)' }} />
                      )}
                    </div>
                    <div style={{ flex: 1, minWidth: 0 }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', gap: 8 }}>
                        <span
                          style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}
                        >
                          {i === 0 ? '最可能' : `第 ${i + 1} 可能`}：{t.disease_cn}
                        </span>
                        <span style={{ flexShrink: 0 }}>{formatPercent(t.prob)}</span>
                      </div>
                      <Progress
                        percent={Math.round((t.prob || 0) * 100)}
                        showInfo={false}
                        strokeColor={i === 0 ? 'var(--c-primary-500)' : 'var(--c-primary-300)'}
                      />
                    </div>
                  </div>
                </div>
              )
            })}
          </Image.PreviewGroup>

          {(knowledge.symptom || knowledge.prevention || knowledge.alias) && (
            <>
              <Typography.Paragraph strong style={{ margin: '12px 0 6px' }}>
                病害说明
              </Typography.Paragraph>
              <Descriptions column={1} size="small" bordered>
                {knowledge.alias && <Descriptions.Item label="又称">{knowledge.alias}</Descriptions.Item>}
                {knowledge.symptom && (
                  <Descriptions.Item label="症状特征">{knowledge.symptom}</Descriptions.Item>
                )}
                {(knowledge.prevention || knowledge.treatment) && (
                  <Descriptions.Item label="防治建议">
                    {knowledge.prevention || knowledge.treatment}
                  </Descriptions.Item>
                )}
              </Descriptions>
            </>
          )}

          <div
            style={{
              marginTop: 16,
              paddingTop: 12,
              borderTop: '1px solid var(--c-border-split)',
              display: 'flex',
              gap: 8,
              flexWrap: 'wrap',
            }}
          >
            <Button icon={<ScissorOutlined />} onClick={() => onRecrop?.(item)}>
              框选单片叶片后重新识别
            </Button>
            {item.record_id && (
              <Button
                type="link"
                icon={<HistoryOutlined />}
                href="/history"
                target="_blank"
                rel="noreferrer"
              >
                在识别历史中查看（#{item.record_id}）
              </Button>
            )}
          </div>
          <Typography.Text type="secondary" style={{ fontSize: 12, display: 'block', marginTop: 8 }}>
            重新识别会<b>新增</b>一条历史记录，原来这条不会被覆盖 —— 这样两版结果都能回看对比。
          </Typography.Text>
        </div>
      </div>
    </Modal>
  )
}
