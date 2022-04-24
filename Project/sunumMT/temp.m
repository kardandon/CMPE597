names = ["ResNet50", "Efficientnet\_b0", "mobilenet\_v3\_small"];
fps = [22.44, 42.97, 85.69];
accuracy = [46.5, 29.38, 34.5];
params = [26, 5.3, 2.49];
colors = ["r","g","b"];

figure();
xlim([0,100]);
ylim([20,50]);
n = ["","",""];
for i=1:3
   plot(fps(i), accuracy(i), colors(i) + "o", "LineWidth", sqrt(params(i))*5, 'MarkerFaceColor', colors(i))
   hold on;
   text(fps(i), accuracy(i)+ 1, names(i))
   n(i) = num2str(params(i));
end
legend(n)
[hleg,icons,plots] = legend('show');
for i=1:3
   ics = findobj(icons, 'MarkerFaceColor', colors(i));
   %ics = findobj(ics, 'Marker', colors(i)+"o");
   set(ics, 'MarkerSize', sqrt(params(i)));
end
title(hleg,'Number of Parameters (M)')
hleg.Title.Visible = 'on';
hleg.FontSize = 22.5;
xlabel("Frame per Second")
ylabel("Accuracy %")
title("PIRL with Different Backbone Architectures")