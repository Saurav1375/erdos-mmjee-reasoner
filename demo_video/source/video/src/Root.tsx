import React from 'react';
import { Composition } from 'remotion';
import { MmjeeDemo, TOTAL, FPS } from './Demo';
import { LiveDemo, LIVE_TOTAL } from './LiveDemo';

export const RemotionRoot: React.FC = () => (
  <>
    <Composition id="MmjeeDemo" component={MmjeeDemo} durationInFrames={TOTAL} fps={FPS} width={1920} height={1080} />
    <Composition id="LiveDemo" component={LiveDemo} durationInFrames={LIVE_TOTAL} fps={FPS} width={1920} height={1080} />
  </>
);
