ps = [0.1,0.5,1,1.5,2,4];

figure('Renderer', 'painters', 'Position', [10 10 800 300])

subplot(1,2,1)
xs = -2:0.01:2;
u = @(x) 1./(1+exp(-10.*x));
plot(xs,u(xs),'linewidth',1)
xlabel('x', 'fontsize',15)
ylabel('f(x)','fontsize',15, 'Rotation',360)

subplot(1,2,2)
for i=1:6
    p0 = ps(i);
    u = @(p0,x) x.^p0.*(x>0)+(-x).^p0.*(x<=0);
    xs = -2:0.01:2;
    plot(xs,u(p0,xs),'linewidth',1)
    ylim([0,2])
    hold on
end
xlabel('x','fontsize',15);
ylabel('|x|^p','fontsize',15, 'Rotation',360);
legend('p=0.1','p=0.5','p=1','p=1.5','p=2','p=4')