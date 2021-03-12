function InitFig(hObject,handles)
clc;
axes(handles.axes1); cla reset; axis on; box on;
set(gca, 'XTickLabel', '', 'YTickLabel', '', 'Color', [0.8039 0.8784 0.9686]);
axes(handles.axes2); cla reset; axis on; box on;
set(gca, 'XTickLabel', '', 'YTickLabel', '', 'Color', [0.8039 0.8784 0.9686]);
set(handles.textInfo, 'String', ...
    '图像去雾系统，首先载入图像并显示，然后选择去雾算法，最后可以观察直方图对比效果。');