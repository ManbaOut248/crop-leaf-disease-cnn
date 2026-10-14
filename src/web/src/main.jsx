// 应用入口：挂载 Ant Design 中文语言包 +「田野翠绿」主题
// 主题设计依据：docs/02_UIUX设计.md V2.0
import React from 'react'
import ReactDOM from 'react-dom/client'
import { ConfigProvider, App as AntApp } from 'antd'
import zhCN from 'antd/locale/zh_CN'
import 'antd/dist/reset.css'
import App from './App'
import './index.css'

/**
 * 主题 Token —— 全站颜色的唯一来源
 *
 * 关键约定（避免可访问性陷阱）：
 *   品牌亮色 #2E9E5B 对白底仅 3.41:1，**只用于图形**（Logo / 图标 / 图表 / 大色块）；
 *   交互主色 #24814A 对白底 4.87:1，满足 WCAG AA，**用于按钮 / 链接 / 选中文字**。
 */
const theme = {
  token: {
    // ---- 品牌与语义 ----
    colorPrimary: '#24814A',
    colorSuccess: '#24814A',
    colorWarning: '#F2A93B',
    colorError: '#D6453E',
    colorInfo: '#3A7BD5',

    // ---- 文本 ----
    colorTextBase: '#2E3A33',
    colorTextHeading: '#1C2620',
    colorTextSecondary: '#6B7A72',
    colorTextTertiary: '#9AA8A0',
    colorTextQuaternary: '#CBD4CE',

    // ---- 背景与边框 ----
    colorBgBase: '#FFFFFF',
    colorBgLayout: '#F7FAF8',
    colorBgContainer: '#FFFFFF',
    colorBorder: '#E2E8E4',
    colorBorderSecondary: '#EFF3F0',

    // ---- 形状与排版 ----
    borderRadius: 8,
    borderRadiusLG: 12,
    fontSize: 14,
    fontFamily:
      '-apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif',
    wireframe: false,
    controlHeight: 36,
  },
  components: {
    Layout: {
      headerBg: '#FFFFFF',
      headerHeight: 64,
      siderBg: '#F7FAF8',
      bodyBg: '#F7FAF8',
    },
    Menu: {
      itemSelectedBg: '#E8F6EE',
      itemSelectedColor: '#1C663A',
      itemHoverBg: '#F1FAF5',
      itemBorderRadius: 8,
      itemHeight: 44,
    },
    Card: {
      borderRadiusLG: 12,
      paddingLG: 24,
      headerFontSize: 16,
    },
    Button: {
      controlHeight: 36,
      primaryShadow: 'none',
      defaultShadow: 'none',
      fontWeight: 500,
    },
    Table: {
      headerBg: '#EFF3F0',
      headerColor: '#1C2620',
      rowHoverBg: '#F1FAF5',
      borderColor: '#EFF3F0',
      headerSplitColor: 'transparent',
    },
    Input: {
      activeShadow: '0 0 0 3px rgba(46, 158, 91, 0.18)',
      paddingBlock: 6,
    },
    Select: { optionSelectedBg: '#E8F6EE' },
    Tag: { borderRadiusSM: 6, defaultBg: '#EFF3F0', defaultColor: '#4A5850' },
    Tabs: { itemSelectedColor: '#1C663A', itemHoverColor: '#24814A', inkBarColor: '#24814A' },
    Progress: { defaultColor: '#2E9E5B', remainingColor: '#EFF3F0' },
    Modal: { borderRadiusLG: 12 },
    Drawer: { paddingLG: 20 },
  },
}

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <ConfigProvider locale={zhCN} theme={theme}>
      <AntApp>
        <App />
      </AntApp>
    </ConfigProvider>
  </React.StrictMode>,
)
