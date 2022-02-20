function Gaussian_Plot(mean_1, std_1, mean_2, std_2)
    x1 = linspace(mean_1 - 5 * std_1, mean_1 + 5 * std_1, 200);
    y1 = normpdf(x1,mean_1,std_1);
    
    plot(x1,y1)
    hold on
    grid on
    
    x2 = linspace(mean_2 - 5 * std_2, mean_2 + 5 * std_2, 200);
    y2 = normpdf(x2,mean_2,std_2);
    plot(x2,y2)
end