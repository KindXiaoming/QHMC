function [qf,pf] = leap_frog_mult(u_grad0,m,q,p,step,L)
p = cellfun(@minus,p,cellfun(@(x) x*step ,u_grad0(q{:}),'un',0),'un',0);
%p = p - step * u_grad(q{:})/2;
for i=1:L
    q = cellfun(@plus,q,cellfun(@(x) x*m^(-1) * step ,p,'un',0),'un',0);
    %q = q + m^(-1) * step * p;
    if i~=L
        p = cellfun(@minus,p,cellfun(@(x) x*step ,u_grad0(q{:}),'un',0),'un',0);
        %p = p - step * u_grad(q{:});
    end
end
p = cellfun(@minus,p,cellfun(@(x) x*step ,u_grad0(q{:}),'un',0),'un',0);
%p = p - step * u_grad(q{:})/2;
qf = q;
pf = p;
end

