<script setup lang="ts">
import { computed } from 'vue'
import type { SseEventPayload } from '@/types/api'

const props = defineProps<{
  events: SseEventPayload[]
}>()

// 只筛选需要展示的事件
const displayEvents = computed(() => {
  return props.events.filter((event) => {
    if (!event.message) return false
    return [
      'job_created',
      'round_started',
      'node_started',
      'planner_started',
      'planner_tool_called',
      'planner_tool_resolved',
      'planner_tool_finished',
      'planner_tool_failed',
      'planner_round_finished',
      'segmentation_started',
      'segmentation_finished',
      'segmentation_failed',
      'package_started',
      'package_finished',
      'package_failed',
      'node_failed',
      'interrupt',
      'job_failed',
    ].includes(event.event)
  })
})

const currentEvent = computed(() => {
  if (displayEvents.value.length === 0) return null
  return displayEvents.value[displayEvents.value.length - 1]
})
</script>

<template>
  <div class="process-viewer glass-panel">
    <div class="orb-container">
      <div class="breathing-orb">
        <div class="orb-content" v-if="currentEvent">
          <transition name="fade-slide" mode="out-in">
            <div :key="currentEvent.message" class="text-wrapper">
              <div class="meta" v-if="currentEvent.round || currentEvent.stage || currentEvent.op">
                <span class="stage-badge" v-if="currentEvent.round">{{ currentEvent.round }}</span>
                <span class="stage-badge" v-if="currentEvent.stage">{{ currentEvent.stage }}</span>
                <span class="op-name" v-if="currentEvent.op">{{ currentEvent.op }}</span>
                <span class="stage-badge" v-if="typeof currentEvent.provider === 'string'">{{ currentEvent.provider }}</span>
              </div>
              <p class="message">{{ currentEvent.message }}</p>
              <p v-if="typeof currentEvent.prompt === 'string'" class="prompt-line">
                分割目标：{{ currentEvent.prompt }}
              </p>
              <p v-if="typeof currentEvent.negative_prompt === 'string'" class="prompt-line muted">
                排除区域：{{ currentEvent.negative_prompt }}
              </p>
              <pre v-if="currentEvent.error" class="error-log">{{ currentEvent.error }}</pre>
            </div>
          </transition>
        </div>
        <div class="orb-content empty" v-else>
          <div class="spinner-pulse-small"></div>
          <p>准备就绪...</p>
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.process-viewer {
  padding: 40px;
  width: 100%;
  display: flex;
  justify-content: center;
  align-items: center;
  min-height: 380px;
}

.orb-container {
  position: relative;
  display: flex;
  justify-content: center;
  align-items: center;
  width: 100%;
  height: 100%;
}

.breathing-orb {
  width: 280px;
  height: 280px;
  border-radius: 50%;
  /* 更柔和通透的玻璃渐变基底，完全去除边界感 */
  background: radial-gradient(circle at 40% 40%, rgba(99, 102, 241, 0.15), rgba(30, 27, 75, 0.4) 70%, transparent 100%);
  display: flex;
  justify-content: center;
  align-items: center;
  /* 更精细的纯粹光晕：多层大半径扩散柔光 */
  box-shadow: 
    0 0 60px rgba(99, 102, 241, 0.3),
    0 0 120px rgba(99, 102, 241, 0.2),
    inset 0 0 60px rgba(99, 102, 241, 0.3);
  animation: breathe 5s cubic-bezier(0.4, 0, 0.2, 1) infinite;
  backdrop-filter: blur(25px);
  -webkit-backdrop-filter: blur(25px);
  /* 移除实线边框，改为极细微内反射高光 */
  border: 1px solid rgba(255, 255, 255, 0.05);
  padding: 30px;
  text-align: center;
  position: relative;
}

.orb-content {
  position: relative;
  z-index: 1;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  width: 100%;
}

.text-wrapper {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 16px;
}

.meta {
  display: flex;
  gap: 10px;
  align-items: center;
  justify-content: center;
  flex-wrap: wrap;
}

.stage-badge {
  font-size: 0.75rem;
  padding: 4px 12px;
  border-radius: 20px;
  background: rgba(255, 255, 255, 0.1);
  border: 1px solid rgba(255, 255, 255, 0.15);
  color: rgba(255, 255, 255, 0.9);
  text-transform: uppercase;
  letter-spacing: 0.5px;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
}

.op-name {
  font-family: 'SF Mono', Monaco, monospace;
  font-size: 0.75rem;
  color: #c7d2fe;
  background: rgba(0, 0, 0, 0.3);
  padding: 3px 10px;
  border-radius: 6px;
}

.message {
  margin: 0;
  font-size: 1.15rem;
  color: #ffffff;
  font-weight: 500;
  line-height: 1.4;
  text-shadow: 0 0 10px rgba(99, 102, 241, 0.8), 0 2px 6px rgba(0, 0, 0, 0.5);
  word-break: break-word;
  animation: pulse-text 5s cubic-bezier(0.4, 0, 0.2, 1) infinite;
}

.prompt-line {
  margin: 0;
  font-size: 0.84rem;
  line-height: 1.45;
  color: rgba(224, 231, 255, 0.92);
  max-width: 220px;
  word-break: break-word;
}

.prompt-line.muted {
  color: rgba(191, 219, 254, 0.72);
}

.error-log {
  margin: 0;
  width: 100%;
  max-width: 220px;
  padding: 10px 12px;
  border-radius: 10px;
  background: rgba(127, 29, 29, 0.3);
  border: 1px solid rgba(239, 68, 68, 0.2);
  color: #fecaca;
  font-size: 0.76rem;
  line-height: 1.45;
  text-align: left;
  white-space: pre-wrap;
  word-break: break-word;
}

.empty p {
  color: rgba(255, 255, 255, 0.6);
  font-size: 1rem;
  margin: 0;
}

.spinner-pulse-small {
  width: 14px;
  height: 14px;
  background-color: rgba(255, 255, 255, 0.5);
  border-radius: 50%;
  animation: pulse-small 1.5s cubic-bezier(0.4, 0, 0.6, 1) infinite;
  margin-bottom: 12px;
}

/* Animations */
/* 更拟人化的呼吸停顿感，缓慢吸气膨胀 -> 快速呼气收缩 */
@keyframes breathe {
  0% { 
    transform: scale(0.98); 
    box-shadow: 
      0 0 40px rgba(99, 102, 241, 0.2),
      0 0 80px rgba(99, 102, 241, 0.1),
      inset 0 0 40px rgba(99, 102, 241, 0.2);
  }
  50% { 
    transform: scale(1.04); 
    box-shadow: 
      0 0 80px rgba(99, 102, 241, 0.45),
      0 0 150px rgba(129, 140, 248, 0.25),
      inset 0 0 80px rgba(129, 140, 248, 0.4);
  }
  100% { 
    transform: scale(0.98); 
    box-shadow: 
      0 0 40px rgba(99, 102, 241, 0.2),
      0 0 80px rgba(99, 102, 241, 0.1),
      inset 0 0 40px rgba(99, 102, 241, 0.2);
  }
}

/* 文字的发光度跟随呼吸轻微变化，增强生命感 */
@keyframes pulse-text {
  0%, 100% { opacity: 0.85; text-shadow: 0 0 8px rgba(99, 102, 241, 0.5); }
  50% { opacity: 1; text-shadow: 0 0 15px rgba(129, 140, 248, 0.9); }
}

@keyframes pulse-small {
  0%, 100% { opacity: 1; transform: scale(1); }
  50% { opacity: 0.3; transform: scale(1.5); }
}

/* 文本淡入淡出滑动切换动效 */
.fade-slide-enter-active,
.fade-slide-leave-active {
  transition: all 0.6s cubic-bezier(0.4, 0, 0.2, 1);
}

.fade-slide-enter-from {
  opacity: 0;
  transform: translateY(12px) scale(0.97);
}

.fade-slide-leave-to {
  opacity: 0;
  transform: translateY(-12px) scale(0.97);
}
</style>
