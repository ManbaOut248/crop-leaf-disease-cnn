// 前端 mock 数据（脚手架 / 演示用）。USE_MOCK=true 时启用。
// 38 类严格对齐后端 src/config.py 的 CLASS_NAMES_ZH（按英文文件夹名字典序 id 0..37）。

const HEALTHY_SYMPTOM = '叶片色泽正常，无病斑、霉层、锈点或畸形，长势良好。'
const HEALTHY_PREVENTION = '保持田间通风透光，合理密植与施肥；定期巡查，发现病叶及时清除并带出田外。'

const RAW = [
  ['苹果', '黑星病', false, '叶片正面出现黑色圆形绒状病斑，果实表面产生黑褐色霉层，影响光合与果品。', '冬季清园剪除病枝病叶；萌芽前喷石硫合剂；生长期交替使用保护性/治疗性杀菌剂。'],
  ['苹果', '黑腐病', false, '果实出现黑褐色凹陷病斑并腐烂，枝条皮层变褐枯死。', '及时摘除病果病枝；加强树体管理增强抗病力；雨季前后喷施咪鲜胺等药剂。'],
  ['苹果', '雪松锈病', false, '叶片产生橙黄色锈状孢子堆，严重时叶片早落。', '清除附近雪松等转主寄主；发病初期喷施三唑类杀菌剂。'],
  ['苹果', '健康', true, HEALTHY_SYMPTOM, HEALTHY_PREVENTION],
  ['蓝莓', '健康', true, HEALTHY_SYMPTOM, HEALTHY_PREVENTION],
  ['樱桃', '白粉病', false, '叶片、嫩梢覆盖白色粉状物，影响光合作用与新梢生长。', '加强通风透光；发病初期喷硫磺制剂或醚菌酯。'],
  ['樱桃', '健康', true, HEALTHY_SYMPTOM, HEALTHY_PREVENTION],
  ['玉米', '灰斑病', false, '叶片出现灰色长条形病斑，严重时整叶枯死。', '选用抗病品种；合理密植；大喇叭口期至抽雄期喷施苯醚甲环唑。'],
  ['玉米', '普通锈病', false, '叶片产生黄褐色夏孢子堆，表皮破裂散出锈粉。', '增施磷钾肥；发病初期喷施三唑酮或戊唑醇。'],
  ['玉米', '北方叶枯病', false, '叶片出现灰绿色水渍状长斑，后变灰白色枯死。', '轮作倒茬；选用抗病种；发病期喷施嘧菌酯。'],
  ['玉米', '健康', true, HEALTHY_SYMPTOM, HEALTHY_PREVENTION],
  ['葡萄', '黑腐病', false, '果实、叶片出现黑色病斑并凹陷，后期产生小黑粒点。', '清除病残体；萌芽期与幼果期喷施保护性杀菌剂。'],
  ['葡萄', '埃斯卡病（黑麻疹）', false, '叶片出现红斑或黄化，枝条维管束变褐，植株衰弱。', '使用无病苗木；避免伤口感染；增施有机肥改良土壤。'],
  ['葡萄', '叶枯病（房枯病）', false, '叶片自边缘向内枯焦，果穗穗轴干枯。', '清除落叶落穗；雨季前喷施多菌灵或代森锰锌。'],
  ['葡萄', '健康', true, HEALTHY_SYMPTOM, HEALTHY_PREVENTION],
  ['柑橘', '黄龙病', false, '新梢叶片斑驳黄化、果实畸形味苦，传播快、危害极大。', '严格检疫无病苗木；及时挖除病树；统防统治木虱。'],
  ['桃', '细菌性斑点病', false, '叶片出现褐色小圆斑并穿孔，果实产生凹陷黑斑。', '选用抗病品种；发芽前喷波尔多液；生长期喷农用链霉素。'],
  ['桃', '健康', true, HEALTHY_SYMPTOM, HEALTHY_PREVENTION],
  ['甜椒', '细菌性斑点病', false, '叶片出现水渍状小斑后变褐穿孔，果实产生隆起黑斑。', '种子消毒；轮作；发病初期喷施铜制剂。'],
  ['甜椒', '健康', true, HEALTHY_SYMPTOM, HEALTHY_PREVENTION],
  ['马铃薯', '早疫病', false, '叶片出现同心轮纹状褐斑，逐渐枯死。', '选用抗病种薯；增施钾肥；发病初期喷代森锰锌或嘧菌酯。'],
  ['马铃薯', '晚疫病', false, '叶片出现水渍状暗绿斑并迅速扩展，潮湿时边缘生白霉。', '选用抗病品种；监测湿度；发病前喷甲霜灵锰锌等。'],
  ['马铃薯', '健康', true, HEALTHY_SYMPTOM, HEALTHY_PREVENTION],
  ['树莓', '健康', true, HEALTHY_SYMPTOM, HEALTHY_PREVENTION],
  ['大豆', '健康', true, HEALTHY_SYMPTOM, HEALTHY_PREVENTION],
  ['南瓜', '白粉病', false, '叶片两面出现白色粉状霉层，后期变灰。', '合理密植通风；发病初期喷醚菌酯或硫磺。'],
  ['草莓', '叶焦病', false, '叶片边缘出现紫褐色焦枯，严重时整叶枯死。', '避免过量钾肥；保持湿度稳定；及时清除病叶。'],
  ['草莓', '健康', true, HEALTHY_SYMPTOM, HEALTHY_PREVENTION],
  ['番茄', '细菌性斑点病', false, '叶片、果实出现褐色小斑点，后期穿孔。', '种子消毒；轮作；发病初期喷农用链霉素或铜制剂。'],
  ['番茄', '早疫病', false, '叶片出现同心轮纹褐斑，茎秆产生椭圆形凹陷斑。', '轮作；增施磷钾肥；发病初期喷代森锰锌。'],
  ['番茄', '晚疫病', false, '叶片出现水渍状暗斑，潮湿时生白霉，果实呈褐色硬块。', '降低田间湿度；发病前喷甲霜灵锰锌或烯酰吗啉。'],
  ['番茄', '叶霉病', false, '叶片背面产生灰紫色霉层，正面变黄。', '加强通风降湿；发病初期喷腐霉利或嘧霉胺。'],
  ['番茄', 'Septoria 叶斑病', false, '叶片出现白色小斑带黑点，逐步扩大枯死。', '清除病叶；轮作；发病期喷施多菌灵。'],
  ['番茄', '红蜘蛛（二斑叶螨）', false, '叶片正面出现黄白色小点，背面有细网与螨体，严重时卷叶。', '保持田间湿度；释放捕食螨；喷施阿维菌素或哒螨灵。'],
  ['番茄', '靶斑病', false, '叶片出现黄褐色圆形靶状斑，中央有孔。', '合理密植；发病初期喷苯醚甲环唑或戊唑醇。'],
  ['番茄', '黄化曲叶病毒病', false, '新叶黄化卷曲、植株矮化，由烟粉虱传播。', '防虫网阻隔烟粉虱；选用抗病品种；及时拔除病株。'],
  ['番茄', '花叶病毒病', false, '叶片出现黄绿相间的花叶、皱缩畸形。', '种子消毒；防蚜虫；及时拔除病株。'],
  ['番茄', '健康', true, HEALTHY_SYMPTOM, HEALTHY_PREVENTION],
]

export const mockDiseases = RAW.map(([crop, disease, isHealthy, symptom, prevention], id) => ({
  id,
  crop,
  disease,
  name: isHealthy ? `${crop}（健康）` : `${crop}${disease}`,
  isHealthy,
  symptom,
  prevention,
}))

// 模拟一次识别结果：以番茄晚疫病为例（Top-3）
export function mockPredict() {
  return {
    top1: {
      class_id: 30,
      name: '番茄晚疫病',
      confidence: 0.918,
      symptom: mockDiseases[30].symptom,
      prevention: mockDiseases[30].prevention,
      crop: '番茄',
      is_healthy: false,
    },
    top3: [
      { class_id: 30, name: '番茄晚疫病', confidence: 0.918 },
      { class_id: 29, name: '番茄早疫病', confidence: 0.054 },
      { class_id: 37, name: '番茄（健康）', confidence: 0.021 },
    ],
    model_version: 'resnet18-plantvillage-38-v1',
    is_mock: true,
  }
}

export function mockRecords() {
  return [
    { id: 1, image_name: 'leaf_001.jpg', top1_name: '番茄晚疫病', top1_confidence: 0.918, created_at: '2026-10-10 10:30:12' },
    { id: 2, image_name: 'leaf_002.jpg', top1_name: '苹果黑星病', top1_confidence: 0.873, created_at: '2026-10-10 09:15:43' },
    { id: 3, image_name: 'leaf_003.jpg', top1_name: '柑橘黄龙病', top1_confidence: 0.901, created_at: '2026-10-09 18:02:55' },
    { id: 4, image_name: 'leaf_004.jpg', top1_name: '马铃薯早疫病', top1_confidence: 0.842, created_at: '2026-10-09 16:41:09' },
  ]
}
