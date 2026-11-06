// 病害参考图路径（2026-10-30 从 DetectPage 抽出，供单张结果与批量结果共用）。
//
// 图片由 scripts/build_disease_refs.py 从 PlantVillage 测试集按分位抽样生成，
// 每类 3 张，落在 src/web/public/disease-ref/<class_id>-<n>.webp。
// 用 class_id 直接拼静态路径 —— 不走后端接口、不需要鉴权头，也不放大 data/ 的攻击面。
// 3 张合计约 3.5MB，但 <img loading="lazy"> 只在真正展示时才下载。
export const REF_PER_CLASS = 3

export function diseaseRefs(classId) {
  // NaN / null / undefined 都视为「没有 class_id」（mock 数据或旧记录），返回空数组，
  // 由调用方渲染占位块 —— 拼出 /disease-ref/undefined-1.webp 会得到一个破图。
  if (classId === null || classId === undefined || Number.isNaN(Number(classId))) return []
  return Array.from({ length: REF_PER_CLASS }, (_, i) => `/disease-ref/${classId}-${i + 1}.webp`)
}
